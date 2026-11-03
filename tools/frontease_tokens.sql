-- LLM tokens stored in frontEASE (Management > Tokens) of one user, for the Task creation scripts.
-- One line per token and connector type: connector_type<TAB>name<TAB>token, by priority. Variable: email.
SELECT ct."ConnectorType", t."Name", t."Token"
FROM "Data"."UserPreferenceTokens" t
JOIN "Data"."UserPreferenceTokenConnectorTypes" ct ON ct."TokenOptionID" = t."ID"
JOIN "Auth"."Users" u ON u."UserPreferencesID" = t."UserPreferencesID"
WHERE u."NormalizedEmail" = upper(:'email')
ORDER BY ct."ConnectorType", t."Priority", t."Name";
