/**
 * Re-simulation driver for full-fidelity replay POV reconstruction (M6 v2).
 *
 * A public Showdown replay omits the private `|request|` stream, so v1 ingestion
 * (data/ingest.py) could only see each player's *revealed* state -- the own bench
 * stays hidden until a mon switches in, an obs distribution the live agent never
 * faces. The replay JSON, however, carries `inputlog`: the PRNG seed plus every
 * choice both players made. Re-driving that through the vendored simulator
 * deterministically reproduces the battle AND re-emits each side's private
 * `|request|`, giving a POV identical in distribution to live play.
 *
 * This is a long-lived filter: it reads NDJSON replays from stdin
 *   {"id": "...", "inputlog": ">start {...}\n>player p1 {...}\n>p1 move ...\n..."}
 * and writes one NDJSON result line per replay
 *   {"id": "...", "p1": [event...], "p2": [event...]}        (success)
 *   {"id": "...", "error": "..."}                            (one bad replay)
 * where each event is {"t":"msg","d":"<chunk>"} (poke-env-ready message lines, in
 * arrival order) or {"t":"choice","d":"move psyblade"} (the inputlog choice that
 * answered the immediately-preceding request -- an explicit decision marker, so the
 * Python side never has to re-derive the per-side choice ordering itself).
 *
 * Driving must be REACTIVE: feeding the whole inputlog at once coalesces requests
 * (BattleStream's sendUpdates keeps only each side's latest pending request), so we
 * answer one request at a time exactly like sim/examples/battle-stream-example.ts.
 *
 * Usage: node resim_driver.js <path-to-pokemon-showdown> < replays.ndjson
 * Requires `<showdown>/dist/sim` to exist (run scripts/setup_server.sh once).
 */

"use strict";

const path = require("path");

const showdownDir = process.argv[2];
if (!showdownDir) {
	process.stderr.write("usage: node resim_driver.js <pokemon-showdown-dir>\n");
	process.exit(2);
}

let BattleStream, getPlayerStreams;
try {
	({ BattleStream, getPlayerStreams } = require(path.resolve(showdownDir, "dist", "sim")));
} catch (e) {
	process.stderr.write(`cannot load ${showdownDir}/dist/sim -- build pokemon-showdown first: ${e}\n`);
	process.exit(2);
}

const REQUEST_PREFIX = "|request|";

/** Split an inputlog into the game spec, per-side choice FIFOs, and terminal controls. */
function parseInputlog(inputlog) {
	const spec = [];
	const queue = { p1: [], p2: [] };
	const control = [];
	for (const line of inputlog.split("\n")) {
		if (line.startsWith(">start ") || line.startsWith(">player ")) {
			spec.push(line);
		} else if (line.startsWith(">p1 ")) {
			queue.p1.push(line.slice(4));
		} else if (line.startsWith(">p2 ")) {
			queue.p2.push(line.slice(4));
		} else if (
			line.startsWith(">forcewin") ||
			line.startsWith(">forcelose") ||
			line.startsWith(">forcetie") ||
			line.startsWith(">tiebreak")
		) {
			control.push(line);
		}
		// `>version` and anything else is informational -- ignore.
	}
	return { spec, queue, control };
}

/** Re-simulate one replay, returning {p1: [event...], p2: [event...]}. */
async function processReplay(inputlog) {
	const { spec, queue, control } = parseInputlog(inputlog);
	const streams = getPlayerStreams(new BattleStream({ noCatch: true }));
	const events = { p1: [], p2: [] };
	let ended = false;

	// Terminate the battle once a side runs out of inputlog choices: flush the
	// recorded `>force*`/`>tiebreak` controls (a forfeit/timeout), or force a tie
	// as a last resort so a truncated inputlog can't hang the drive loops.
	function endBattle() {
		if (ended) return;
		ended = true;
		if (control.length) {
			for (const line of control) void streams.omniscient.write(line);
		} else {
			void streams.omniscient.write(">forcetie");
		}
	}

	async function drive(side) {
		for await (const chunk of streams[side]) {
			events[side].push({ t: "msg", d: chunk });
			for (const line of chunk.split("\n")) {
				if (!line.startsWith(REQUEST_PREFIX)) continue;
				const payload = line.slice(REQUEST_PREFIX.length);
				if (!payload) continue;
				let request;
				try {
					request = JSON.parse(payload);
				} catch {
					continue;
				}
				if (!request || request.wait) continue; // opponent-only turn: no choice
				const choice = queue[side].shift();
				if (choice !== undefined) {
					events[side].push({ t: "choice", d: choice });
					void streams[side].write(choice);
				} else {
					endBattle(); // forfeit/timeout point: no sample is recorded here
				}
			}
		}
	}

	const done = Promise.all([drive("p1"), drive("p2")]);
	void streams.omniscient.write(spec.join("\n"));
	await done;
	return events;
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
			const events = await processReplay(String(replay.inputlog || ""));
			process.stdout.write(JSON.stringify({ id, p1: events.p1, p2: events.p2 }) + "\n");
		} catch (e) {
			process.stdout.write(JSON.stringify({ id, error: String(e && e.stack || e) }) + "\n");
		}
	}
}

main().catch((e) => {
	process.stderr.write(`fatal: ${e && e.stack || e}\n`);
	process.exit(1);
});
