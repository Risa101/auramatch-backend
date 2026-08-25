-- 2026-08-26_widen_analysis_history_image_path.sql
--
-- Fixes: POST /api/save-analysis returning 500 on every real request.
--
-- The frontend sends the uploaded photo's preview as a base64 data URL
-- (e.g. "data:image/jpeg;base64,...", tens of KB of text) in the
-- `image_path` field. The column was `varchar(255)`, so MySQL rejected
-- every insert with:
--   (1406, "Data too long for column 'image_path' at row 1")
--
-- Already applied to the local dev DB directly; this file is what to run
-- against the production database during deploy. Safe / non-destructive —
-- widening a column, no data loss, no downtime.
--
-- Run manually against production, e.g.:
--   mysql -h <host> -u <user> -p <database> < migrations/2026-08-26_widen_analysis_history_image_path.sql
-- or paste the ALTER statement into Railway's DB console.

ALTER TABLE `analysis_history` MODIFY COLUMN `image_path` LONGTEXT DEFAULT NULL;
