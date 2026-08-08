"""Version constants that participate in content hashing.

Any change to these invalidates previously-built indexes, which is the point:
an index is only comparable to another built by the same code. Bump the
relevant constant whenever its subsystem's output could change.
"""

# Bump when the ingest/normalisation pipeline changes the text it indexes.
#
# 1.1.0 (M1.2) - the build now tokenises documents and writes an inverted
#   index. The indexed document is `name + description`; `type` and
#   `source_ref` are deliberately excluded, see retrieval/tokenize.py.
#   This necessarily changes every index's content_hash, which is correct:
#   the index contains strictly more than it did at 1.0.0.
PARSER_VERSION = "1.1.0"

# Bump when scoring, the sort key, or quantisation changes.
RANKER_VERSION = "1.0.0"

# Bump when the content-addressed ID recipe changes. This invalidates every
# node_id and edge_id in existence, so treat it as a migration.
ID_SCHEMA_VERSION = 1

# Manifest format version.
MANIFEST_VERSION = 1

# The released version. Bumped on every push; recorded in spec/frozen.json by
# `drf freeze`, so a tag names an exact spec, index and result set rather than
# just a commit.
RELEASE_VERSION = "0.0.4"
