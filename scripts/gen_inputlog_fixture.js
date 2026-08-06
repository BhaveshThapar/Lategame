/**
 * Regenerate the gen9randombattle inputlog fixture that `tests/conftest.py` holds.
 *
 * WHY THIS EXISTS. The R-PREDICT fidelity test and the resim end-to-end test both replay a
 * hardcoded inputlog: a fixed PRNG seed plus a fixed list of choices. That is only replayable
 * against the simulator rev that produced it -- gen9randombattle SETS CHANGE UPSTREAM, so the
 * same seed rolls different teams under a different rev and the recorded choices go illegal
 * partway through. When `SHOWDOWN_REV` in scripts/setup_server.sh is bumped, the fixture is
 * invalidated and must be regenerated HERE, in the same commit as the bump.
 *
 * Two RandomPlayerAIs play a full battle to a natural `|win|`, so the log is legal by
 * construction (rather than transcribed from a replay, which is how the previous fixture came
 * to encode a rev nobody recorded). Prints the inputlog to stdout.
 *
 * Usage: node scripts/gen_inputlog_fixture.js [showdown-dir] [seed]
 */

"use strict";

const path = require("path");

const showdownDir = process.argv[2] || "third_party/pokemon-showdown";
// The battle seed. Fixed by default so this script is itself reproducible; override to search
// for a longer battle when a regenerated one comes up too short for the tests' turn thresholds.
const seed = process.argv[3] || "sodium,b8493732a42936a1fd687ddf7988dbbc";

let BattleStream, getPlayerStreams, RandomPlayerAI;
try {
	({ BattleStream, getPlayerStreams } = require(
		path.resolve(showdownDir, "dist", "sim", "battle-stream.js"),
	));
	({ RandomPlayerAI } = require(
		path.resolve(showdownDir, "dist", "sim", "tools", "random-player-ai.js"),
	));
} catch (e) {
	process.stderr.write(
		`cannot load ${showdownDir}/dist/sim -- build pokemon-showdown first ` +
			`(bash scripts/setup_server.sh): ${e}\n`,
	);
	process.exit(2);
}

async function main() {
	const stream = new BattleStream();
	const streams = getPlayerStreams(stream);

	// Every seed is pinned. The per-player seed drives TEAM GENERATION and is separate from the
	// battle seed and from the AI choice seeds -- omit it and the stream invents one per run, so
	// two invocations of this script would emit two different fixtures. All four are fixed so
	// regenerating against an unchanged simulator rev is a no-op.
	const spec = { formatid: "gen9randombattle", seed };
	const p1spec = { name: "gummiworm", seed: "sodium,a3572989c38af097ba39a7fa38627767" };
	const p2spec = { name: "Qwuartz", seed: "sodium,d9f6a625ddf93230a09fa7bc6f0239d0" };

	// Distinct AI seeds so the two sides do not make correlated choices.
	new RandomPlayerAI(streams.p1, { seed: "sodium,5c1a3f0e9d2b8746af03c5e17b9d4028" }).start();
	new RandomPlayerAI(streams.p2, { seed: "sodium,e64b90d7c3821fa50d6e4b17293af8c1" }).start();

	void streams.omniscient.write(
		`>start ${JSON.stringify(spec)}\n` +
			`>player p1 ${JSON.stringify(p1spec)}\n` +
			`>player p2 ${JSON.stringify(p2spec)}`,
	);

	// Drain to completion; the AIs answer every request off their own streams.
	for await (const chunk of streams.omniscient) void chunk;

	const battle = stream.battle;
	if (!battle) {
		process.stderr.write("battle did not initialize\n");
		process.exit(1);
	}
	if (!battle.ended) {
		process.stderr.write("battle did not reach a winner -- try another seed\n");
		process.exit(1);
	}
	process.stderr.write(`[gen_inputlog] turns=${battle.turn} winner=${battle.winner}\n`);
	process.stdout.write(battle.inputLog.join("\n") + "\n");
}

main().catch((e) => {
	process.stderr.write(String((e && e.stack) || e) + "\n");
	process.exit(1);
});
