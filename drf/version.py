"""Version constants that participate in content hashing.

Any change to these invalidates previously-built indexes, which is the point:
an index is only comparable to another built by the same code. Bump the
relevant constant whenever its subsystem's output could change.
"""

# Bump when the ingest/normalisation pipeline changes the text it indexes.
PARSER_VERSION = "1.0.0"

# Bump when scoring, the sort key, or quantisation changes.
RANKER_VERSION = "1.0.0"

# Bump when the content-addressed ID recipe changes. This invalidates every
# node_id and edge_id in existence, so treat it as a migration.
ID_SCHEMA_VERSION = 1

# Manifest format version.
MANIFEST_VERSION = 1
