# A standing "challenge RotomAI on Showdown" service.
#
# This image runs the LIVE CLIENT only. It deliberately does not carry node or the vendored
# simulator: `third_party/pokemon-showdown` is the local `--no-security` sim used for training and
# gating, and live play connects to the public server instead. That keeps the image small enough
# for the smallest VPS tier the agent will actually fit on.
#
#   docker build -t rotomai .
#   docker run -d --restart=unless-stopped \
#     -e ROTOMAI_PS_USERNAME=YourBotAccount \
#     -e ROTOMAI_PS_PASSWORD=... \
#     -v "$PWD/results:/app/results" \
#     rotomai
#
# SIZE THE BOX FOR THE INSTALL, NOT THE RUN. The agent itself is 4.56M parameters and idles in tens
# of MB, but pip's peak RSS while unpacking the torch wheel will OOM a 512 MB machine. 1 GB is the
# floor; the built image is ~700 MB, nearly all of it torch.
FROM python:3.11-slim

WORKDIR /app

# CPU wheels only. Without the index override pip resolves the CUDA build -- about 2 GB of driver
# payload for a network that has never once run on a GPU (docs/OPERATIONS.md, "No GPU").
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-live.txt ./
RUN pip install --no-cache-dir -r requirements-live.txt

COPY pyproject.toml README.md LICENSE ./
COPY rotomai ./rotomai
RUN pip install --no-cache-dir --no-deps -e .

# The released gen9ou policy. Not baked in: `checkpoints/` is gitignored and the weights live on the
# GitHub release, so they are fetched at build time and verified against the committed manifest
# rather than trusted. See README "Using the released weights".
ARG RELEASE=v1.0.0
ADD https://github.com/BhaveshThapar/RotomAI/releases/download/${RELEASE}/iter_320.pt \
    /app/checkpoints/ppo_v26b_s0/iter_320.pt

# `--mode accept` accepts opt-in challenges only and is NOT the policy-gated ladder path: `--mode
# ladder` additionally requires `--ladder-ack` and ROTOMAI_LIVE_ALLOW_LADDER, neither of which this
# image sets. `--battle-delay 10` and the default concurrency of 1 are the etiquette knobs; poke-env
# reads concurrency 0 as UNLIMITED, which is the visible signature of farming, and the client
# refuses it.
ENTRYPOINT ["python", "-m", "rotomai.cli", "live"]
CMD ["--mode", "accept", \
     "--n", "0", \
     "--agent", "offrl", \
     "--checkpoint", "checkpoints/ppo_v26b_s0/iter_320.pt", \
     "--format", "gen9ou", \
     "--concurrency", "1", \
     "--battle-delay", "10", \
     "--out-dir", "results/accept"]
