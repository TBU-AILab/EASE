#!/bin/sh
# Creates smoke Tasks (tools/create_smoke_tasks.py) or the complete paper experiment
# (tools/create_experiment_tasks.py) with the LLM tokens stored in frontEASE
# (Management > Tokens) of the given user. Run on the server in the frontEASE directory (with docker-compose.yml):
#
#   src/FoP_IMT.Core/tools/create_tasks_frontease.sh you@example.com --provider openai --summarizer-provider cerit
#   src/FoP_IMT.Core/tools/create_tasks_frontease.sh you@example.com --experiment
#
# All arguments after the optional --experiment are passed to the selected script (--author is set to the e-mail).
# The tokens go from the postgres container to the core container through a pipe, so they never appear
# on a command line or in the shell history. Every connector of a Task gets the token stored for its connector
# type. Smoke selection uses --token <name> / --summarizer-token <name>;
# experiment selection uses repeatable --token <provider>=<name>.
set -eu

if [ $# -lt 1 ] || [ -z "$1" ]; then
    echo "Usage: $0 <frontEASE user e-mail> [--experiment] [script options]" >&2
    exit 2
fi
EMAIL="$1"
shift

SCRIPT="create_smoke_tasks.py"
if [ "${1:-}" = "--experiment" ]; then
    SCRIPT="create_experiment_tasks.py"
    shift
fi

# one line per token and connector type: connector_type<TAB>name<TAB>token (by priority)
docker compose exec -T postgres sh -c \
    'psql -U "$POSTGRES_USER" -d FrontEASE -X -q -A -t -F "$(printf "\t")" -v ON_ERROR_STOP=1 -v email="$1" -f -' \
    sh "$EMAIL" <<'SQL' | docker compose exec -T backend-core python "tools/$SCRIPT" --tokens-stdin --author "$EMAIL" "$@"
SELECT ct."ConnectorType", t."Name", t."Token"
FROM "Data"."UserPreferenceTokens" t
JOIN "Data"."UserPreferenceTokenConnectorTypes" ct ON ct."TokenOptionID" = t."ID"
JOIN "Auth"."Users" u ON u."UserPreferencesID" = t."UserPreferencesID"
WHERE u."NormalizedEmail" = upper(:'email')
ORDER BY ct."ConnectorType", t."Priority", t."Name";
SQL
