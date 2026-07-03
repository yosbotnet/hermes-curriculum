-- Motivation layer: telemetry, question kill switch, edge provenance.
ALTER TABLE question ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'active';
ALTER TABLE edge ADD COLUMN IF NOT EXISTS provenance text NOT NULL DEFAULT 'inferred'
    CHECK (provenance IN ('spine', 'inferred', 'manual'));
ALTER TABLE edge ADD COLUMN IF NOT EXISTS confidence real NOT NULL DEFAULT 0.6;
CREATE TABLE IF NOT EXISTS engagement_log (
    id      bigserial PRIMARY KEY,
    kind    text NOT NULL CHECK (kind IN ('check', 'escalate', 'session_start', 'session_end', 'item_flag')),
    course  text NOT NULL,
    at      timestamptz NOT NULL DEFAULT now(),
    payload jsonb NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS engagement_course_idx ON engagement_log (course, kind, at DESC);
