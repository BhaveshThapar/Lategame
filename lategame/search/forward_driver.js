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

const DEFAULT_FORMAT = "gen9randombattle";

// FORMAT IS PER-SPEC, NOT PER-PROCESS, and the random-set generator is RB-ONLY.
//
// Only Random Battles have a `Teams.getGenerator(...).randomSet` to invent a legal set from a bare
// species. On a TEAMBUILT format (gen9ou, gen9vgc*) that generator either does not exist or answers
// for a different format's legality, so calling it would silently fabricate illegal sets -- the
// caller supplies complete sets instead (see determinize.battle_to_spec, which fills unrevealed
// fields from the Smogon usage prior).
//
// Cached per format id: `Dex.forFormat` and the generator are not free, and one agent's driver
// serves thousands of reconstructions on the same format.
const _cache = new Map();
function forFormat(formatid) {
	const id = toID(formatid) || DEFAULT_FORMAT;
	let entry = _cache.get(id);
	if (!entry) {
		const dex = Dex.forFormat(id);
		let generator = null;
		let allSpecies = [];
		if (id.includes("randombattle")) {
			try {
				generator = Teams.getGenerator(id);
				allSpecies = Object.keys(generator.randomSets || {});
			} catch {
				generator = null;
			}
		}
		entry = { id, dex, generator, allSpecies };
		_cache.set(id, entry);
	}
	return entry;
}

const idOf = (x) => (x && typeof x === "object" && "id" in x ? x.id : x || "");
const toID = (s) => String(s || "").toLowerCase().replace(/[^a-z0-9]/g, "");

/** Build a legal set for `species`, overriding with observed fields.
 *
 * Random Battles: start from the format's own generated set and keep observed fields.
 * Teambuilt: there is no generator, so `info` must already carry the full set -- anything the
 * caller left unspecified falls back to a neutral, legal-by-construction spread rather than to
 * another format's idea of a legal set. */
function buildSet(info, rng, fmt) {
	const sp = fmt.dex.species.get(info.species);
	const neutral = {
		species: sp.name, name: sp.name, level: info.level || 100, moves: [],
		ability: sp.abilities?.["0"] || "", item: "",
		evs: { hp: 85, atk: 85, def: 85, spa: 85, spd: 85, spe: 85 },
		ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 },
		gender: "", teraType: sp.types?.[0] || "Normal",
	};
	let base = neutral;
	if (fmt.generator) {
		try {
			base = fmt.generator.randomSet(sp, {}, false, false);
		} catch {
			base = neutral;
		}
	}
	const set = { ...base, species: sp.name, name: sp.name };
	const moves = (info.moves || []).map(toID).filter(Boolean);
	if (moves.length) {
		// Keep observed moves; top up from the base set's legal picks to 4.
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
	if (info.nature) set.nature = info.nature;
	if (info.evs) set.evs = { ...set.evs, ...info.evs };
	if (info.ivs) set.ivs = { ...set.ivs, ...info.ivs };
	return set;
}

/** Order the team so the active mon is slot 1 (active after >start); sample opp fill. */
function buildTeam(sideSpec, rng, fmt) {
	const sets = [];
	const used = new Set();
	for (const info of sideSpec.team) {
		sets.push(buildSet(info, rng, fmt));
		used.add(toID(info.species));
	}
	// Filling unknown SPECIES only makes sense where the opponent's roster is hidden -- i.e.
	// Random Battles. On a teambuilt format team preview names all six up front, so the caller
	// sends fill=0 and what is unknown is the SETS, which it supplies from the usage prior.
	const fill = fmt.allSpecies.length ? sideSpec.fill || 0 : 0;
	let guard = 0;
	while (sets.length < 6 && fill > 0 && guard++ < 200) {
		const cand = fmt.allSpecies[Math.floor(rng() * fmt.allSpecies.length)];
		if (used.has(cand)) continue;
		used.add(cand);
		sets.push(buildSet({ species: cand }, rng, fmt));
		if (sets.length >= 6) break;
	}
	// Move the active mon(s) to the front so they lead. Doubles has TWO, in slot order.
	return reorderActiveFirst(sets, activeIndices(sideSpec));
}

/** `active` is an int on singles and a list on doubles; normalize to an ordered index list. */
function activeIndices(sideSpec) {
	const a = sideSpec.active;
	if (Array.isArray(a)) return a.filter((i) => Number.isInteger(i) && i >= 0);
	return [Number.isInteger(a) ? a : 0];
}

/** Reorder so `indices` occupy the leading slots, preserving their given order. */
function reorderActiveFirst(items, indices) {
	const lead = [];
	const taken = new Set();
	for (const i of indices) {
		if (i < items.length && !taken.has(i)) {
			lead.push(items[i]);
			taken.add(i);
		}
	}
	const rest = items.filter((_, i) => !taken.has(i));
	return [...lead, ...rest];
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

/** Per-slot legal choices: the moves one active can make (with targets) plus its switches. */
function slotChoices(side, slotIndex, doubles) {
	const out = [];
	const active = side.active[slotIndex];
	if (active && !active.fainted) {
		const slots = active.moveSlots || [];
		for (let i = 0; i < slots.length; i++) {
			const m = slots[i];
			if (m.disabled) continue;
			if ((m.pp ?? 1) <= 0) continue;
			if (!doubles) {
				out.push({ choice: `move ${i + 1}`, type: "move", id: toID(m.id), slot: slotIndex });
				continue;
			}
			// DOUBLES MOVES NEED A TARGET, and which targets are legal is a property of the move:
			// spread/self/ally-only moves take no explicit foe. Ask the simulator rather than
			// assuming -- a target Showdown rejects costs the whole turn.
			const move = side.battle.dex.moves.get(m.id);
			const targets = [];
			if (move && ["normal", "any", "adjacentFoe"].includes(move.target)) {
				for (let t = 1; t <= 2; t++) {
					const foe = side.foe.active[t - 1];
					if (foe && !foe.fainted) targets.push(t);
				}
			}
			if (!targets.length) targets.push(null); // no explicit target (spread, self, status)
			for (const t of targets) {
				out.push({
					choice: t === null ? `move ${i + 1}` : `move ${i + 1} ${t}`,
					type: "move",
					id: toID(m.id),
					slot: slotIndex,
					target: t,
				});
			}
		}
	}
	for (let j = 0; j < side.pokemon.length; j++) {
		const p = side.pokemon[j];
		if (side.active.indexOf(p) >= 0 || p.fainted) continue;
		out.push({
			choice: `switch ${j + 1}`,
			type: "switch",
			id: idOf(p.species),
			slot: slotIndex,
			switchIndex: j + 1,
		});
	}
	return out;
}

/** Legal choices for a side, read from the live (post-edit) battle objects + labelled.
 *
 * Singles returns one entry per action. DOUBLES returns one entry per JOINT action -- the cross
 * product of the two slots' choices, minus the combinations Showdown would reject. The joint form
 * is what `battle.choose` takes ("move 1 1, move 2 2"), and it is also the honest branching factor
 * for search: the two slots are not independent decisions. */
function legalChoices(side) {
	const doubles = (side.active || []).length > 1;
	if (!doubles) return slotChoices(side, 0, false);

	const first = slotChoices(side, 0, true);
	const second = slotChoices(side, 1, true);
	// A slot with no active (fainted, awaiting a replacement) contributes a literal "pass".
	const pass = [{ choice: "pass", type: "pass", id: "", slot: -1 }];
	const left = first.length ? first : pass;
	const right = second.length ? second : pass;

	const out = [];
	for (const a of left) {
		for (const b of right) {
			// BOTH SLOTS MAY NOT SWITCH TO THE SAME BENCHED POKEMON. Showdown rejects the order
			// outright and plays a default move instead, which is a silent lost turn.
			if (a.type === "switch" && b.type === "switch" && a.switchIndex === b.switchIndex) {
				continue;
			}
			out.push({
				choice: `${a.choice}, ${b.choice}`,
				type: a.type === "switch" && b.type === "switch" ? "switch" : "move",
				id: `${a.id}+${b.id}`,
				parts: [a, b],
			});
		}
	}
	return out;
}

/** A `|request|` line for `side` with each mon's condition patched to the post-edit hp/status.
 * applyState mutates hp/status directly (no protocol), so the raw activeRequest is stale;
 * poke-env reads a side's own team hp from its request, so we correct it here (Lever 14). */
function stateRequest(side) {
	const r = side.activeRequest;
	if (!r || r.wait) return null;
	const clone = JSON.parse(JSON.stringify(r));
	const mons = clone.side && Array.isArray(clone.side.pokemon) ? clone.side.pokemon : [];
	for (let k = 0; k < side.pokemon.length && k < mons.length; k++) {
		const mon = side.pokemon[k];
		mons[k].condition = mon.fainted
			? "0 fnt"
			: `${mon.hp}/${mon.maxhp}${mon.status ? " " + mon.status : ""}`;
	}
	return "|request|" + JSON.stringify(clone);
}

/** Step a freshly started battle past TEAM PREVIEW, which teambuilt formats have and RB does not.
 *
 * On gen9randombattle `>start` lands directly on turn 1 with both leads active. Every teambuilt
 * format (gen9ou, VGC) opens on a team-preview request instead, where NOTHING is active -- so a
 * reconstruction stops there, `legalChoices` finds no active Pokemon and reports only switches,
 * and search would evaluate a position in which neither side can move. Silent, and it looks like
 * "the opponent has no moves" rather than like a phase error.
 *
 * `buildTeam` has already put the observed active in slot 1, so selecting the identity order
 * preserves it. `maxTeamSize` is what makes this correct for VGC, where team preview BRINGS 6 and
 * PICKS 4 -- picking all six would be rejected. */
function advancePastTeamPreview(stream, battle) {
	const needs = (r) => r && r.teamPreview && !r.wait;
	for (let guard = 0; guard < 4; guard++) {
		if (!battle.sides.some((s) => needs(s.activeRequest))) return;
		for (let i = 0; i < battle.sides.length; i++) {
			const side = battle.sides[i];
			if (!needs(side.activeRequest)) continue;
			const size = Math.min(
				side.activeRequest.maxTeamSize || side.pokemon.length,
				side.pokemon.length,
			);
			const order = Array.from({ length: size }, (_, k) => k + 1).join("");
			stream.write(`>p${i + 1} team ${order}`);
		}
	}
}

function reconstruct(spec) {
	const rng = mulberry32((spec.seed || 0) >>> 0);
	// Build teams; record the active-first reordering so per-slot state lines up.
	for (const key of ["p1", "p2"]) {
		const ss = spec[key];
		// idx[newSlot] = originalIndex; the same reordering buildTeam applies, so the per-slot
		// state list lines up with the team the simulator actually received.
		ss._order = reorderActiveFirst(
			ss.team.map((_, j) => j),
			activeIndices(ss),
		);
	}
	const fmt = forFormat(spec.format);
	const t1 = Teams.pack(buildTeam(spec.p1, rng, fmt));
	const t2 = Teams.pack(buildTeam(spec.p2, rng, fmt));
	const stream = new BattleStream({ noCatch: true });
	stream.write(
		`>start {"formatid":"${fmt.id}"}\n` +
			`>player p1 {"name":"p1","team":"${t1}"}\n` +
			`>player p2 {"name":"p2","team":"${t2}"}`,
	);
	const battle = stream.battle;
	advancePastTeamPreview(stream, battle);
	applyState(battle, spec);
	// Per-side POV logs (Lever 14): channel-filter the omniscient init log so a fresh poke-env
	// Battle can be built from the opponent's perspective (leads revealed, own team full).
	const ch = extractChannelMessages(battle.log.join("\n"), [1, 2]);
	return {
		state: State.serializeBattle(battle),
		digest: digest(battle),
		p1_choices: legalChoices(battle.sides[0]),
		p2_choices: legalChoices(battle.sides[1]),
		p1_log: (ch[1] || []).join("\n"),
		p2_log: (ch[2] || []).join("\n"),
		p1_request: stateRequest(battle.sides[0]),
		p2_request: stateRequest(battle.sides[1]),
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
		// The same observable digest `reconstruct` returns, now also at the post-step node.
		//
		// It is here so a SHAPED leaf can be scored without building a poke-env Battle at all.
		// `data.reward.state_value` reads only per-mon hp / fainted / status, the two team sizes
		// and the winner -- every one of which is in this digest. The alternative path (deepcopy
		// the live battle, feed the delta line by line, then read the same fields off it) measured
		// ~170 ms per node against ~1 ms for the RPC, which is what made a depth-2 VGC search cost
		// ~190 s/battle and put M2 out of reach.
		digest: digest(fork),
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
