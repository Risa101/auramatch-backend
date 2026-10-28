-- 2026-08-30_add_product_platforms.sql
--
-- Adds multi-platform shopping links to products (Shopee / Lazada / TikTok
-- Shop, extensible to future platforms without another schema change).
--
-- Context: `products.links` (exposed to the frontend as `shop_url`) only
-- ever held ONE url per product, so a product could only point to a single
-- storefront. This adds a child table so one product can list a URL per
-- platform, while leaving `products.links` untouched for backward
-- compatibility with existing consumers.
--
-- Safe / additive / non-destructive:
--   - No existing table is altered, dropped, or has data removed.
--   - `product_platforms` is a brand new table.
--   - `platform` is VARCHAR (not ENUM) so a new platform can be added later
--     by just inserting rows — no further ALTER TABLE needed.
--   - FK to products(product_id) ON DELETE CASCADE only matters for hard
--     deletes; product_service.delete_product() soft-deletes (sets
--     deleted_at) and never actually removes a row, so this is a defensive
--     safety net, not something that fires in normal operation.
--
-- Already applied to the local dev DB directly; this file is what to run
-- against the production database during deploy, e.g.:
--   mysql -h <host> -u <user> -p <database> < migrations/2026-08-30_add_product_platforms.sql

CREATE TABLE IF NOT EXISTS `product_platforms` (
  `id`             INT(11)      NOT NULL AUTO_INCREMENT,
  `product_id`     VARCHAR(50)  NOT NULL,
  `platform`       VARCHAR(30)  NOT NULL,       -- e.g. 'shopee', 'lazada', 'tiktok_shop'
  `product_url`    TEXT         NOT NULL,
  `affiliate_url`  TEXT         DEFAULT NULL,   -- reserved for future affiliate-link support
  `is_active`      TINYINT(1)   NOT NULL DEFAULT 1,
  `created_at`     TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at`     TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_product_platform` (`product_id`, `platform`),
  KEY `idx_product_platforms_product` (`product_id`),
  KEY `idx_product_platforms_active` (`is_active`),
  CONSTRAINT `fk_product_platforms_product`
    FOREIGN KEY (`product_id`) REFERENCES `products` (`product_id`)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
