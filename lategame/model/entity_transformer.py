"""Per-entity transformer actor-critic (plan.md, M5).

Drop-in replacement for the flat ``ActorCritic`` MLP. The M2 encoder already
emits a flat ``obs`` that is a deterministic concatenation of fixed-size blocks
(``features.encoder.OBS_LAYOUT``): 6 ego + 6 opponent Pokemon, 4 active moves,
1 global block. This model slices that flat vector **back into per-entity tokens**
(no re-encoding, so every existing shard/dataset is reused), projects each token
type to ``d_model``, adds learned per-position embeddings, and runs self-attention
so the policy can reason *relationally* (my speed vs theirs, my move's damage vs
their HP) -- the thing the bag-of-mons MLP could not represent.

The critic is **decoupled** (two-tower): actor and critic share the token
embedding but have separate transformer trunks, separate ``[CLS]`` pooling tokens,
and the critic has a deeper value MLP. This targets the M4 bottleneck (weak
critic, value MAE ~1.9) without the actor and critic fighting over one shared
representation.

The value head is unchanged in *semantics* -- logits over fixed return bins, read
back via ``value_from_logits`` -- so ``value_support`` / ``hl_gauss_target`` /
``value_from_logits`` and the self-play value-support pinning are reused as-is.
Forward contract matches ``ActorCritic``: ``(obs) -> (policy_logits, value_logits)``.
"""

from __future__ import annotations

import torch
from torch import nn

from lategame.features.action_space import GEN9_ACTION_SPACE_SIZE
from lategame.features.encoder import (
    MOVE_ID_FIELDS,
    OBS_DIM,
    OBS_LAYOUT,
    POKEMON_ID_FIELDS,
    ObsLayout,
)
from lategame.features.vocab import vocab_sizes


def _make_encoder(
    d_model: int, n_heads: int, ff_dim: int, n_layers: int, dropout: float
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=ff_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    # norm_first disables the nested-tensor fast path anyway; set it explicitly to
    # avoid a noisy UserWarning.
    return nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)


def _flat_dim(layout: ObsLayout) -> int:
    """Flat observation width implied by a token layout."""
    return layout.global_start + layout.global_dim


def _layout_for(input_dim: int) -> ObsLayout:
    """The layout a flat width implies -- singles unless it is the doubles width.

    Keyed on the dim rather than carried in the checkpoint because every checkpoint written before
    doubles existed stores only `input_dim`; resolving here means none of them need rewriting.
    """
    from lategame.features.doubles_encoder import OBS_LAYOUT_DOUBLES

    return OBS_LAYOUT_DOUBLES if input_dim == _flat_dim(OBS_LAYOUT_DOUBLES) else OBS_LAYOUT


class EntityTransformer(nn.Module):
    """Entity-token transformer with a decoupled (two-tower) critic."""

    def __init__(
        self,
        input_dim: int = OBS_DIM,
        n_actions: int = GEN9_ACTION_SPACE_SIZE,
        n_bins: int = 51,
        dropout: float = 0.1,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_dim: int = 256,
        id_embed: bool = True,
        id_embed_dim: int = 32,
        id_embed_init: str = "random",
        layout: ObsLayout | None = None,
    ) -> None:
        super().__init__()
        # The token layout is a PARAMETER, defaulting to singles. G4 asks for a third format
        # "without rewriting the core", and this is the seam: doubles is the same architecture over
        # a layout with two active slots (21 tokens, 888-d) rather than one (17 tokens, 761-d).
        # Resolved from `input_dim` when not given, so every existing checkpoint -- which stores
        # only input_dim -- keeps building exactly the model it always did.
        layout = layout or _layout_for(input_dim)
        if input_dim != _flat_dim(layout):
            raise ValueError(
                f"EntityTransformer expects flat obs of {_flat_dim(layout)} for the given layout, "
                f"got {input_dim}."
            )
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.id_embed = id_embed
        self.id_embed_dim = id_embed_dim
        self.id_embed_init = id_embed_init

        self._layout = layout
        self._n_tokens = layout.n_tokens

        # Identity embeddings (R-ENCODE): learn a vector per species/move/item/ability so
        # the policy can reason about *which* entity, not just its numerics. ``id_embed``
        # off is the controlled ablation -- the trailing ID channels are dropped and only
        # the hand-crafted numerics feed the trunk (the pre-R-ENCODE encoder).
        if id_embed:
            sizes = vocab_sizes()
            # Prefix keys: ``nn.ModuleDict`` reserves names like ``items`` (a dict method).
            self.embeddings = nn.ModuleDict(
                {
                    f"emb_{cat}": nn.Embedding(sizes[cat], id_embed_dim, padding_idx=0)
                    for cat in (*POKEMON_ID_FIELDS, *MOVE_ID_FIELDS)
                }
            )
            poke_in = layout.pokemon_numeric_dim + len(POKEMON_ID_FIELDS) * id_embed_dim
            move_in = layout.move_numeric_dim + len(MOVE_ID_FIELDS) * id_embed_dim
        else:
            self.embeddings = nn.ModuleDict()
            poke_in = layout.pokemon_numeric_dim
            move_in = layout.move_numeric_dim

        # Shared token embedding: one projection per token *type*, plus a learned
        # per-position embedding (subsumes role + slot for the fixed layout).
        self.poke_proj = nn.Linear(poke_in, d_model)
        self.move_proj = nn.Linear(move_in, d_model)
        self.global_proj = nn.Linear(layout.global_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, layout.n_tokens, d_model))
        self.input_dropout = nn.Dropout(dropout)

        # Actor tower.
        self.actor_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.actor_encoder = _make_encoder(d_model, n_heads, ff_dim, n_layers, dropout)
        self.policy_head = nn.Linear(d_model, n_actions)

        # Critic tower (decoupled): own CLS, own trunk, deeper value MLP.
        self.critic_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.critic_encoder = _make_encoder(d_model, n_heads, ff_dim, n_layers, dropout)
        self.value_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.value_head = nn.Linear(d_model, n_bins)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.actor_cls, std=0.02)
        nn.init.normal_(self.critic_cls, std=0.02)
        if self.id_embed and self.id_embed_init == "prior":
            self._apply_id_priors()

    @torch.no_grad()
    def _apply_id_priors(self) -> None:
        """Warm-start species/move embeddings from dex-feature priors (R-ENCODE).

        Overwritten by ``load_state_dict`` when a trained checkpoint is loaded, so this
        only seeds *fresh* models; items/abilities (no priors) keep their random init.
        """
        from lategame.features.embed_prior import load_id_priors

        for cat, weight in load_id_priors(self.id_embed_dim).items():
            emb = self.embeddings[f"emb_{cat}"]
            assert isinstance(emb, nn.Embedding)
            emb.weight.copy_(torch.from_numpy(weight))

    def arch_config(self) -> dict[str, int | bool | str]:
        """Architecture hyperparameters for the checkpoint ``arch`` block."""
        return {
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "ff_dim": self.ff_dim,
            "id_embed": self.id_embed,
            "id_embed_dim": self.id_embed_dim,
            "id_embed_init": self.id_embed_init,
        }

    def _entity_features(
        self, block: torch.Tensor, numeric_dim: int, id_fields: tuple[str, ...]
    ) -> torch.Tensor:
        """Split ``[numeric | ids]`` and replace each trailing ID channel with its embedding."""
        numeric = block[..., :numeric_dim]
        if not self.id_embed:
            return numeric
        feats = [numeric]
        for offset, cat in enumerate(id_fields):
            ids = block[..., numeric_dim + offset].long()
            feats.append(self.embeddings[f"emb_{cat}"](ids))
        return torch.cat(feats, dim=-1)

    def _tokenize(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Flat obs -> (tokens (B, n_tokens, d_model), padding_mask (B, n_tokens))."""
        layout = self._layout
        batch = obs.shape[0]

        poke_flat = obs[:, : layout.moves_start].reshape(
            batch, layout.n_pokemon_tokens, layout.pokemon_dim
        )
        move_flat = obs[:, layout.moves_start : layout.global_start].reshape(
            batch, layout.n_move_blocks, layout.move_dim
        )
        global_flat = obs[:, layout.global_start :]

        poke_tok = self.poke_proj(
            self._entity_features(poke_flat, layout.pokemon_numeric_dim, POKEMON_ID_FIELDS)
        )
        move_tok = self.move_proj(
            self._entity_features(move_flat, layout.move_numeric_dim, MOVE_ID_FIELDS)
        )
        global_tok = self.global_proj(global_flat).unsqueeze(1)
        tokens = torch.cat([poke_tok, move_tok, global_tok], dim=1) + self.pos_embed
        tokens = self.input_dropout(tokens)

        # "present" flag is feature 0 of each Pokemon/move block; global is always
        # present. Padding mask is True where a key should be ignored (absent).
        poke_absent = poke_flat[:, :, 0] <= 0.5
        move_absent = move_flat[:, :, 0] <= 0.5
        global_absent = torch.zeros(batch, 1, dtype=torch.bool, device=obs.device)
        padding_mask = torch.cat([poke_absent, move_absent, global_absent], dim=1)
        return tokens, padding_mask

    def _tower(
        self,
        encoder: nn.TransformerEncoder,
        cls: torch.Tensor,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run one tower and return the pooled ``[CLS]`` representation."""
        batch = tokens.shape[0]
        cls_tok = cls.expand(batch, -1, -1)
        x = torch.cat([cls_tok, tokens], dim=1)
        cls_keep = torch.zeros(batch, 1, dtype=torch.bool, device=tokens.device)
        mask = torch.cat([cls_keep, padding_mask], dim=1)
        x = encoder(x, src_key_padding_mask=mask)
        return x[:, 0]

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, padding_mask = self._tokenize(obs)
        actor_repr = self._tower(self.actor_encoder, self.actor_cls, tokens, padding_mask)
        critic_repr = self._tower(self.critic_encoder, self.critic_cls, tokens, padding_mask)
        policy_logits = self.policy_head(actor_repr)
        value_logits = self.value_head(self.value_mlp(critic_repr))
        return policy_logits, value_logits
