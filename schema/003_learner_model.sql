-- Learner model (dialogue mode): what the tutor learns about the learner.
-- Insights and fixed misconceptions carry FSRS state so they can come back as
-- transfer questions; see docs/tutor-contract.md. Idempotent.
CREATE TABLE IF NOT EXISTS learner_note (
    id            text PRIMARY KEY,
    course        text NOT NULL,
    topic         text NOT NULL DEFAULT '',
    kind          text NOT NULL CHECK (kind IN ('insight', 'misconception_fixed', 'open_thread', 'material_gap', 'goal')),
    text          text NOT NULL,
    learner_words text,
    source_ref    jsonb,
    created_at    timestamptz NOT NULL DEFAULT now(),
    status        text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'resolved')),
    resolved_at   timestamptz,
    stability     real,
    difficulty    real,
    last_review   timestamptz,
    due_at        timestamptz,
    reps          int NOT NULL DEFAULT 0,
    lapses        int NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS learner_note_course_idx ON learner_note (course, kind, status);
CREATE INDEX IF NOT EXISTS learner_note_due_idx ON learner_note (course, due_at) WHERE status = 'active';
