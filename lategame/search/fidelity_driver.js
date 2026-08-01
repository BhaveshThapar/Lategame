/**
 * Forward-model fidelity driver (Lever 11 / R-PREDICT, Gate A -- the cheap KILL gate).
 *
 * R-PREDICT needs a *forward model*: from a battle state, step a hypothetical
 * (my-move, opp-move) -> next state, so depth-1 expectimax can evaluate each of our
 * actions with the GREEN value head. The forward primitive is the vendored simulator's
 * State.serializeBattle / State.deserializeBattle (a fork) plus battle.choose (a step).
 * Before building any search on top of it, we must prove that primitive is FAITHFUL:
 * forking a battle and stepping the fork must reproduce, bit-for-bit, what stepping the
 * battle directly produces. The serialized state carries the PRNG seed, so the two paths
 * draw identical random numbers (damage rolls, crits, accuracy) -- a faithful fork is an
 * *exact* match, not an approximation.
 *
 * This is a long-lived NDJSON filter (mirrors data/resim_driver.js): reads
 *   {"id": "...", "inputlog": ">start {...}\n>p1 move ...\n..."}
 * and, replaying each real replay turn-by-turn at the Battle level, emits
 *   {"id", "transitions", "core_mismatch", "full_mismatch", "drive_errors", "samples": [...]}
 * where `core` = {hp, status, fainted, active, field, hazards, turn} (what the encoder /
 * value fn consume) and `full` additionally pins boosts / item / ability / pp / tera.
 *
 * Usage: node fidelity_driver.js <path-to-pokemon-showdown> < replays.ndjson
 * Requires `<showdown>/dist/sim` to exist (run scripts/setup_server.sh once).
 */

"use strict";

const path = require("path");

const showdownDir = process.argv[2];
if (!showdownDir) {
	process.stderr.write("usage: node fidelity_driver.js <pokemon-showdown-dir>\n");
	process.exit(2);
}

let BattleStream, State;
try {
	({ BattleStream } = require(path.resolve(showdownDir, "dist", "sim", "battle-stream.js")));
	({ State } = require(path.resolve(showdownDir, "dist", "sim", "state.js")));
} catch (e) {
	process.stderr.write(`cannot load ${showdownDir}/dist/sim -- build pokemon-showdown first: ${e}\n`);
	process.exit(2);
}

const MAX_SAMPLES = 3; // first few mismatch diffs per replay, for diagnostics

/** Split an inputlog into the game spec and per-side choice FIFOs (mirrors resim_driver). */
function parseInputlog(inputlog) {
	const spec = [];
	const queue = { p1: [], p2: [] };
	for (const line of inputlog.split("\n")) {
		if (line.startsWith(">start ") || line.startsWith(">player ")) {
			spec.push(line);
		} else if (line.startsWith(">p1 ")) {
			queue.p1.push(line.slice(4));
		} else if (line.startsWith(">p2 ")) {
			queue.p2.push(line.slice(4));
		}
		// `>version`, `>forcewin`, etc. are not replayed: we only score clean turns.
	}
	return { spec, queue };
}

const idOf = (x) => (x && typeof x === "object" && "id" in x ? x.id : x || "");

/** Semantic per-pokemon snapshot. `full` adds fields beyond what search consumes. */
function snapPokemon(side, p, full) {
	const core = {
		species: idOf(p.species),
		hp: p.hp,
		maxhp: p.maxhp,
		status: p.status || "",
		fainted: !!p.fainted,
		active: side.active.indexOf(p) >= 0,
	};
	if (!full) return core;
	const boosts = {};
	for (const k of Object.keys(p.boosts).sort()) boosts[k] = p.boosts[k];
	return {
		...core,
		item: p.item || "",
		ability: p.ability || "",
		tera: p.terastallized || "",
		boosts,
		pp: p.moveSlots.map((m) => [m.id, m.pp]),
	};
}

function snapSide(side, full) {
	const hazards = {};
	for (const k of Object.keys(side.sideConditions).sort()) {
		hazards[k] = side.sideConditions[k].layers ?? 1;
	}
	return {
		pokemon: side.pokemon.map((p) => snapPokemon(side, p, full)),
		hazards,
	};
}

/** Deterministic digest of the fields that matter; `full` is the stricter superset. */
function digest(battle, full) {
	return JSON.stringify({
		turn: battle.turn,
		ended: !!battle.ended,
		winner: battle.winner ?? null,
		weather: idOf(battle.field.weather),
		terrain: idOf(battle.field.terrain),
		pseudo: Object.keys(battle.field.pseudoWeather).sort(),
		sides: battle.sides.map((s) => snapSide(s, full)),
	});
}

/** Sides that must submit a choice this request round (waiters get {wait:true}). */
function pendingSides(battle) {
	const out = [];
	for (const side of battle.sides) {
		const req = side.activeRequest;
		if (req && !req.wait) out.push(side.id);
	}
	return out;
}

/** Re-simulate one replay, forking + comparing at every decision round. */
function processReplay(inputlog) {
	const { spec, queue } = parseInputlog(inputlog);
	const stream = new BattleStream({ noCatch: true });
	stream.write(spec.join("\n")); // synchronous: populates stream.battle + first request
	const battle = stream.battle;
	if (!battle) return { error: "battle did not initialize" };

	let transitions = 0;
	let coreMismatch = 0;
	let fullMismatch = 0;
	let driveErrors = 0;
	const samples = [];

	while (battle.requestState && !battle.ended) {
		const sides = pendingSides(battle);
		if (!sides.length) break;
		const round = [];
		let truncated = false;
		for (const sid of sides) {
			const c = queue[sid].shift();
			if (c === undefined) {
				truncated = true;
				break;
			}
			round.push([sid, c]);
		}
		if (truncated) break; // forfeit / timeout: out of recorded choices

		// Detach a fork seed BEFORE the real battle commits this round. Stringify so the
		// real battle's continued mutation of shared arrays (e.g. log) can't touch it.
		const seed = JSON.stringify(State.serializeBattle(battle));

		let ok = true;
		for (const [sid, c] of round) {
			if (!battle.choose(sid, c)) {
				ok = false;
				break;
			}
		}
		if (!ok) {
			driveErrors++;
			break; // a bad drive desyncs the rest of this replay
		}

		const fork = State.deserializeBattle(seed);
		let forkOk = true;
		for (const [sid, c] of round) {
			if (!fork.choose(sid, c)) {
				forkOk = false;
				break;
			}
		}
		if (!forkOk) {
			driveErrors++;
			if (samples.length < MAX_SAMPLES) {
				samples.push({ turn: battle.turn, round, kind: "fork_choose_failed" });
			}
			continue;
		}

		transitions++;
		const realCore = digest(battle, false);
		const forkCore = digest(fork, false);
		if (realCore !== forkCore) {
			coreMismatch++;
			if (samples.length < MAX_SAMPLES) {
				samples.push({ turn: battle.turn, round, real: JSON.parse(realCore), fork: JSON.parse(forkCore) });
			}
		}
		if (digest(battle, true) !== digest(fork, true)) fullMismatch++;
	}

	return { transitions, core_mismatch: coreMismatch, full_mismatch: fullMismatch, drive_errors: driveErrors, samples };
}

async function main() {
	let input = "";
	process.stdin.setEncoding("utf8");
	for await (const chunk of process.stdin) input += chunk;

	for (const raw of input.split("\n")) {
		const line = raw.trim();
		if (!line) continue;
		let replay;
		try {
			replay = JSON.parse(line);
		} catch (e) {
			process.stdout.write(JSON.stringify({ id: null, error: `bad json: ${e}` }) + "\n");
			continue;
		}
		const id = replay.id ?? null;
		try {
			const res = processReplay(String(replay.inputlog || ""));
			process.stdout.write(JSON.stringify({ id, ...res }) + "\n");
		} catch (e) {
			process.stdout.write(JSON.stringify({ id, error: String((e && e.stack) || e) }) + "\n");
		}
	}
}

main().catch((e) => {
	process.stderr.write(`fatal: ${(e && e.stack) || e}\n`);
	process.exit(1);
});
