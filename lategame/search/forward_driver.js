/**
 * Persistent forward-model driver (Lever 11 / R-PREDICT).
 *
 * Gate A proved the simulator's serialize/fork/step primitive is bit-for-bit faithful.
 * This driver turns it into a forward model for live search. It is a long-lived NDJSON
 * request/response process (one per agent -- node startup is paid once) supporting:
 *
 *   {"cmd":"ping"}                                   -> {"pong":true}
 *   {"cmd":"reconstruct","id":..,"spec":{...}}       -> {"id":..,"state":<serialized>,
 *                                                        "digest":{...}}      (observable digest)
 *   {"cmd":"step","id":..,"state":<serialized>,      -> {"id":..,"state":<serialized'>,
 *        "p1":<choice|null>,"p2":<choice|null>}          "p1_delta":"<lines>","p1_request":<json|null>,
 *                                                        "p1_choices":[..],"p2_choices":[..],
 *                                                        "ended":bool,"winner":<name|null>}
 *
 * `reconstruct` builds a full two-sided battle from a determinization spec (our known team +
 * a sampled opponent), then sets observed scalars (hp fraction, status, boosts, fainted,
 * field, hazards). Hidden opponent details are guesses; the *observable* state is honored, so
 * `digest` (what the reconstruction mini-gate checks) reflects the live POV. `step` forks a
 * serialized state, applies a hypothetical (p1,p2) choice, and returns the clean per-side
 * delta + new request (fog-of-war filtered) so the caller can feed a poke-env Battle and embed
 * the leaf -- the exact path resim uses, validated against deepcopy of the live root.
 *
 * Usage: node forward_driver.js <path-to-pokemon-showdown>
 */

"use strict";

const path = require("path");
const readline = require("readline");

const showdownDir = process.argv[2];
if (!showdownDir) {
	process.stderr.write("usage: node forward_driver.js <pokemon-showdown-dir>\n");
	process.exit(2);
}

let BattleStream, State, extractChannelMessages, Teams, Dex;
try {
	({ BattleStream } = require(path.resolve(showdownDir, "dist", "sim", "battle-stream.js")));
	({ State } = require(path.resolve(showdownDir, "dist", "sim", "state.js")));
	({ extractChannelMessages } = require(path.resolve(showdownDir, "dist", "sim", "battle.js")));
	({ Teams } = require(path.resolve(showdownDir, "dist", "sim", "teams.js")));
	({ Dex } = require(path.resolve(showdownDir, "dist", "sim", "dex.js")));
} catch (e) {
	process.stderr.write(`cannot load ${showdownDir}/dist/sim -- build pokemon-showdown first: ${e}\n`);
	process.exit(2);
}

const FORMAT = "gen9randombattle";
const generator = Teams.getGenerator(FORMAT);
const dex = Dex.forFormat(FORMAT);
const ALL_SPECIES = Object.keys(generator.randomSets || {});

const idOf = (x) => (x && typeof x === "object" && "id" in x ? x.id : x || "");
const toID = (s) => String(s || "").toLowerCase().replace(/[^a-z0-9]/g, "");

/** Build a legal gen9 RB set for `species`, overriding with observed fields. */
function buildSet(info, rng) {
	const sp = dex.species.get(info.species);
	let base;
	try {
		base = generator.randomSet(sp, {}, false, false);
	} catch {
		base = { species: sp.name, name: sp.name, level: 80, moves: [], ability: sp.abilities?.["0"] || "",
			item: "", evs: { hp: 85, atk: 85, def: 85, spa: 85, spd: 85, spe: 85 },
			ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 }, gender: "", teraType: sp.types?.[0] || "Normal" };
	}
	const set = { ...base, species: sp.name, name: sp.name };
	const moves = (info.moves || []).map(toID).filter(Boolean);
	if (moves.length) {
		// Keep observed moves; top up from the generator's legal picks to 4.
		const merged = [...moves];
		for (const m of (base.moves || []).map(toID)) {
			if (merged.length >= 4) break;
			if (!merged.includes(m)) merged.push(m);
		}
		set.moves = merged.slice(0, 4);
	}
	if (info.ability) set.ability = info.ability;
	if (info.item) set.item = info.item;
	if (info.gender) set.gender = info.gender;
	if (info.level) set.level = info.level;
	if (info.teraType) set.teraType = info.teraType;
	return set;
}

/** Order the team so the active mon is slot 1 (active after >start); sample opp fill. */
function buildTeam(sideSpec, rng) {
	const sets = [];
	const used = new Set();
	for (const info of sideSpec.team) {
		sets.push(buildSet(info, rng));
		used.add(toID(info.species));
	}
	const fill = sideSpec.fill || 0;
	let guard = 0;
	while (sets.length < 6 && fill > 0 && guard++ < 200) {
		const cand = ALL_SPECIES[Math.floor(rng() * ALL_SPECIES.length)];
		if (used.has(cand)) continue;
		used.add(cand);
		sets.push(buildSet({ species: cand }, rng));
		if (sets.length >= 6) break;
	}
	// Move the active mon to slot 1 so it leads.
	const a = sideSpec.active || 0;
	if (a > 0 && a < sets.length) {
		const [act] = sets.splice(a, 1);
		sets.unshift(act);
	}
	return sets;
}

/** Mulberry32 PRNG from an int seed -> () -> [0,1). */
function mulberry32(seed) {
	let t = seed >>> 0;
	return function () {
		t += 0x6d2b79f5;
		let x = Math.imul(t ^ (t >>> 15), 1 | t);
		x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
		return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
	};
}

/** Apply observed per-mon + field/hazard scalars onto a freshly started battle. */
function applyState(battle, spec) {
	for (const [i, sideSpec] of [spec.p1, spec.p2].entries()) {
		const side = battle.sides[i];
		// team was reordered active-first, so the per-slot state list is reordered to match.
		const order = sideSpec._order || side.pokemon.map((_, j) => j);
		for (let j = 0; j < side.pokemon.length; j++) {
			const st = (sideSpec.state || [])[order[j]];
			if (!st) continue;
			const mon = side.pokemon[j];
			const frac = st.hp_frac != null ? st.hp_frac : 1;
			if (st.fainted || frac <= 0) {
				mon.hp = 0;
				mon.fainted = true;
			} else {
				mon.hp = Math.max(1, Math.round(frac * mon.maxhp));
			}
			if (st.status) {
				mon.status = toID(st.status);
				mon.statusState = { id: toID(st.status) };
			}
			if (side.active.indexOf(mon) >= 0) {
				// Reset to observed boosts, clearing start-of-battle ability effects (e.g.
				// Intimidate) that >start applies but the real turn-N state doesn't have.
				for (const k of Object.keys(mon.boosts)) mon.boosts[k] = 0;
				if (st.boosts) for (const k of Object.keys(st.boosts)) mon.boosts[k] = st.boosts[k];
			}
		}
		side.pokemonLeft = side.pokemon.filter((p) => !p.fainted).length;
		// Hazards / screens via the proper method (correct internal state).
		const hz = (spec.hazards || {})[i === 0 ? "p1" : "p2"] || {};
		for (const [cond, layers] of Object.entries(hz)) {
			const n = Math.max(1, Number(layers) || 1);
			for (let l = 0; l < n; l++) side.addSideCondition(toID(cond), "debug");
		}
	}
	const f = spec.field || {};
	// Set to observed, CLEARING any weather/terrain a determinized lead's ability (Drought,
	// Grassy Surge, ...) applied at >start but the real turn-N state doesn't have.
	if (f.weather) battle.field.setWeather(toID(f.weather), "debug");
	else {
		battle.field.weather = "";
		battle.field.weatherState = { id: "" };
	}
	if (f.terrain) battle.field.setTerrain(toID(f.terrain), "debug");
	else {
		battle.field.terrain = "";
		battle.field.terrainState = { id: "" };
	}
	for (const pw of f.pseudo || []) battle.field.addPseudoWeather(toID(pw), "debug");
	if (spec.turn) battle.turn = Number(spec.turn);
}

/** Observable digest (what the reconstruction mini-gate compares). */
function digest(battle) {
	const side = (s) => {
		const out = {};
		for (const p of s.pokemon) {
			const isActive = s.active.indexOf(p) >= 0;
			const boosts = {};
			if (isActive) {
				for (const k of Object.keys(p.boosts).sort()) if (p.boosts[k]) boosts[k] = p.boosts[k];
			}
			out[idOf(p.species)] = {
				hp: Math.round((p.hp / (p.maxhp || 1)) * 100) / 100,
				status: p.status || "",
				fainted: !!p.fainted,
				active: isActive,
				boosts,
			};
		}
		return out;
	};
	const hz = (s) => {
		const o = {};
		for (const k of Object.keys(s.sideConditions).sort()) o[k] = s.sideConditions[k].layers ?? 1;
		return o;
	};
	return {
		turn: battle.turn,
		weather: idOf(battle.field.weather),
		terrain: idOf(battle.field.terrain),
		p1: side(battle.sides[0]),
		p2: side(battle.sides[1]),
		hazards: { p1: hz(battle.sides[0]), p2: hz(battle.sides[1]) },
	};
}

/** Legal choices for a side, read from the live (post-edit) battle objects + labelled. */
function legalChoices(side) {
	const out = [];
	const active = side.active[0];
	if (active && !active.fainted) {
		const slots = active.moveSlots || [];
		for (let i = 0; i < slots.length; i++) {
			const m = slots[i];
			if (m.disabled) continue;
			if ((m.pp ?? 1) <= 0) continue;
			out.push({ choice: `move ${i + 1}`, type: "move", id: toID(m.id) });
		}
	}
	for (let j = 0; j < side.pokemon.length; j++) {
		const p = side.pokemon[j];
		if (side.active.indexOf(p) >= 0 || p.fainted) continue;
		out.push({ choice: `switch ${j + 1}`, type: "switch", id: idOf(p.species) });
	}
	return out;
}

function reconstruct(spec) {
	const rng = mulberry32((spec.seed || 0) >>> 0);
	// Build teams; record the active-first reordering so per-slot state lines up.
	for (const key of ["p1", "p2"]) {
		const ss = spec[key];
		const a = ss.active || 0;
		const idx = ss.team.map((_, j) => j);
		if (a > 0 && a < idx.length) {
			const [x] = idx.splice(a, 1);
			idx.unshift(x);
		}
		// idx[newSlot] = originalIndex; invert so order[newSlot] indexes state[].
		ss._order = idx;
	}
	const t1 = Teams.pack(buildTeam(spec.p1, rng));
	const t2 = Teams.pack(buildTeam(spec.p2, rng));
	const stream = new BattleStream({ noCatch: true });
	stream.write(
		`>start {"formatid":"${FORMAT}"}\n` +
			`>player p1 {"name":"p1","team":"${t1}"}\n` +
			`>player p2 {"name":"p2","team":"${t2}"}`,
	);
	const battle = stream.battle;
	applyState(battle, spec);
	return {
		state: State.serializeBattle(battle),
		digest: digest(battle),
		p1_choices: legalChoices(battle.sides[0]),
		p2_choices: legalChoices(battle.sides[1]),
	};
}

function step(serialized, p1, p2) {
	const fork = State.deserializeBattle(typeof serialized === "string" ? serialized : JSON.stringify(serialized));
	const collected = [];
	fork.send = (type, data) => {
		if (type === "update") collected.push(Array.isArray(data) ? data.join("\n") : data);
	};
	fork.sentLogPos = fork.log.length;
	let illegal = false;
	if (p1 && fork.sides[0].activeRequest && !fork.sides[0].activeRequest.wait) {
		if (!fork.choose("p1", p1)) illegal = true;
	}
	if (!illegal && p2 && fork.sides[1].activeRequest && !fork.sides[1].activeRequest.wait) {
		if (!fork.choose("p2", p2)) illegal = true;
	}
	if (illegal) return { illegal: true };
	fork.sendUpdates();
	const full = collected.join("\n");
	const ch = extractChannelMessages(full, [1, 2]);
	const reqLine = (side) => {
		const r = side.activeRequest;
		return r && !r.wait ? "|request|" + JSON.stringify(r) : null;
	};
	return {
		p1_delta: (ch[1] || []).join("\n"),
		p2_delta: (ch[2] || []).join("\n"),
		p1_request: reqLine(fork.sides[0]),
		p2_request: reqLine(fork.sides[1]),
		// Legal choices at the resulting node (driver frame: p1 = us), so depth>1
		// search can recurse without re-parsing the request. Empty for a side that
		// is waiting / has no decision this step.
		p1_choices: legalChoices(fork.sides[0]),
		p2_choices: legalChoices(fork.sides[1]),
		ended: !!fork.ended,
		winner: fork.winner ?? null,
		state: State.serializeBattle(fork),
	};
}

function handle(msg) {
	switch (msg.cmd) {
		case "ping":
			return { pong: true };
		case "reconstruct":
			return { id: msg.id, ...reconstruct(msg.spec) };
		case "step":
			return { id: msg.id, ...step(msg.state, msg.p1, msg.p2) };
		default:
			return { id: msg.id, error: `unknown cmd ${msg.cmd}` };
	}
}

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", (line) => {
	line = line.trim();
	if (!line) return;
	let msg;
	try {
		msg = JSON.parse(line);
	} catch (e) {
		process.stdout.write(JSON.stringify({ error: `bad json: ${e}` }) + "\n");
		return;
	}
	try {
		process.stdout.write(JSON.stringify(handle(msg)) + "\n");
	} catch (e) {
		process.stdout.write(JSON.stringify({ id: msg.id ?? null, error: String((e && e.stack) || e) }) + "\n");
	}
});
