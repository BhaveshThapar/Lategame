"""R-TEAM: team provisioning for teambuilt formats (live play / eval / self-play).

Random Battles get server-generated teams; teambuilt formats (e.g. gen9ou) do not, so
every ``Player`` must supply a legal team. This package holds a poke-env ``Teambuilder``
over a curated, pre-validated team pool. Training data recovers teams from replay logs
instead (see ``lategame.data.resim``); this is not that path.
"""
