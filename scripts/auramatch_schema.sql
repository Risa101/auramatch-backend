CREATE DATABASE IF NOT EXISTS auramatch
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE auramatch;

CREATE TABLE IF NOT EXISTS `user` (
  user_id INT NOT NULL AUTO_INCREMENT,
  username VARCHAR(100) NULL,
  email VARCHAR(191) NOT NULL,
  password VARCHAR(255) NOT NULL,
  avatar TEXT NULL,
  role VARCHAR(32) NOT NULL DEFAULT 'user',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  deleted_at DATETIME NULL,
  PRIMARY KEY (user_id),
  UNIQUE KEY uq_user_email (email),
  UNIQUE KEY uq_user_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS superadmin (
  superadmin_id INT NOT NULL AUTO_INCREMENT,
  username VARCHAR(100) NOT NULL,
  password VARCHAR(255) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (superadmin_id),
  UNIQUE KEY uq_superadmin_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS brand (
  brand_id INT NOT NULL AUTO_INCREMENT,
  brand_name VARCHAR(150) NOT NULL,
  logo_path TEXT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  deleted_at DATETIME NULL,
  PRIMARY KEY (brand_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS products (
  product_id VARCHAR(64) NOT NULL,
  name VARCHAR(255) NOT NULL,
  category VARCHAR(100) NULL,
  brand_id INT NULL,
  image_url TEXT NULL,
  price DECIMAL(10,2) NULL DEFAULT 0.00,
  rating DECIMAL(3,2) NULL DEFAULT 0.00,
  status VARCHAR(32) NOT NULL DEFAULT 'active',
  personal_color_tags VARCHAR(255) NULL,
  finish_type VARCHAR(100) NULL,
  coverage_level VARCHAR(100) NULL,
  suitable_for_skin_type VARCHAR(100) NULL,
  stock INT NULL DEFAULT 0,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  deleted_at DATETIME NULL,
  PRIMARY KEY (product_id),
  KEY idx_products_brand_id (brand_id),
  CONSTRAINT fk_products_brand
    FOREIGN KEY (brand_id) REFERENCES brand (brand_id)
    ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS review (
  review_id INT NOT NULL AUTO_INCREMENT,
  user_id INT NOT NULL,
  product_id VARCHAR(64) NOT NULL,
  rating INT NOT NULL,
  comment TEXT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (review_id),
  KEY idx_review_user_id (user_id),
  KEY idx_review_product_id (product_id),
  CONSTRAINT fk_review_user
    FOREIGN KEY (user_id) REFERENCES `user` (user_id)
    ON DELETE CASCADE,
  CONSTRAINT fk_review_product
    FOREIGN KEY (product_id) REFERENCES products (product_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS favorite (
  favorite_id INT NOT NULL AUTO_INCREMENT,
  user_id INT NOT NULL,
  product_id VARCHAR(64) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted_at DATETIME NULL,
  PRIMARY KEY (favorite_id),
  UNIQUE KEY uq_favorite_user_product (user_id, product_id),
  KEY idx_favorite_product_id (product_id),
  CONSTRAINT fk_favorite_user
    FOREIGN KEY (user_id) REFERENCES `user` (user_id)
    ON DELETE CASCADE,
  CONSTRAINT fk_favorite_product
    FOREIGN KEY (product_id) REFERENCES products (product_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS promotion (
  promotion_id INT NOT NULL AUTO_INCREMENT,
  promo_name VARCHAR(255) NOT NULL,
  promo_detail TEXT NULL,
  brand_id INT NULL,
  discount_percent DECIMAL(5,2) NULL,
  coupon_code VARCHAR(100) NULL,
  min_price DECIMAL(10,2) NULL,
  max_discount DECIMAL(10,2) NULL,
  promo_type VARCHAR(100) NULL,
  season VARCHAR(50) NULL,
  start_date DATETIME NULL,
  end_date DATETIME NULL,
  status VARCHAR(32) NULL DEFAULT 'active',
  logo_url TEXT NULL,
  superadmin_id INT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  deleted_at DATETIME NULL,
  PRIMARY KEY (promotion_id),
  KEY idx_promotion_brand_id (brand_id),
  KEY idx_promotion_superadmin_id (superadmin_id),
  CONSTRAINT fk_promotion_brand
    FOREIGN KEY (brand_id) REFERENCES brand (brand_id)
    ON DELETE SET NULL,
  CONSTRAINT fk_promotion_superadmin
    FOREIGN KEY (superadmin_id) REFERENCES superadmin (superadmin_id)
    ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS analysis_history (
  history_id INT NOT NULL AUTO_INCREMENT,
  user_id INT NOT NULL,
  season VARCHAR(50) NULL,
  face_shape VARCHAR(50) NULL,
  eyebrows VARCHAR(100) NULL,
  eyes VARCHAR(100) NULL,
  nose VARCHAR(100) NULL,
  lips VARCHAR(100) NULL,
  image_path TEXT NULL,
  score INT NOT NULL DEFAULT 100,
  analysis_date DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (history_id),
  KEY idx_analysis_history_user_id (user_id),
  CONSTRAINT fk_analysis_history_user
    FOREIGN KEY (user_id) REFERENCES `user` (user_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS password_resets (
  reset_id INT NOT NULL AUTO_INCREMENT,
  user_id INT NOT NULL,
  token_hash VARCHAR(255) NOT NULL,
  expires_at DATETIME NOT NULL,
  used_at DATETIME NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (reset_id),
  UNIQUE KEY uq_password_resets_token_hash (token_hash),
  KEY idx_password_resets_user_id (user_id),
  CONSTRAINT fk_password_resets_user
    FOREIGN KEY (user_id) REFERENCES `user` (user_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS profiles (
  user_id INT NOT NULL,
  display_name VARCHAR(150) NULL,
  gender VARCHAR(50) NULL,
  birthdate DATE NULL,
  bio TEXT NULL,
  aura_color VARCHAR(100) NULL,
  skin_tone VARCHAR(100) NULL,
  location VARCHAR(150) NULL,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  deleted_at DATETIME NULL,
  PRIMARY KEY (user_id),
  CONSTRAINT fk_profiles_user
    FOREIGN KEY (user_id) REFERENCES `user` (user_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS user_photos (
  photo_id INT NOT NULL AUTO_INCREMENT,
  user_id INT NOT NULL,
  photo_url TEXT NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (photo_id),
  KEY idx_user_photos_user_id (user_id),
  CONSTRAINT fk_user_photos_user
    FOREIGN KEY (user_id) REFERENCES `user` (user_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS looks (
  look_id INT NOT NULL AUTO_INCREMENT,
  look_name VARCHAR(150) NOT NULL,
  personal_color VARCHAR(50) NOT NULL,
  image_url TEXT NULL,
  status VARCHAR(32) NOT NULL DEFAULT 'active',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (look_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS youtube (
  youtube_id INT NOT NULL AUTO_INCREMENT,
  video_url TEXT NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  deleted_at DATETIME NULL,
  PRIMARY KEY (youtube_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS stock (
  stock_id INT NOT NULL AUTO_INCREMENT,
  product_id VARCHAR(64) NOT NULL,
  quantity INT NOT NULL DEFAULT 0,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (stock_id),
  UNIQUE KEY uq_stock_product_id (product_id),
  CONSTRAINT fk_stock_product
    FOREIGN KEY (product_id) REFERENCES products (product_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `order` (
  order_id INT NOT NULL AUTO_INCREMENT,
  user_id INT NULL,
  status VARCHAR(32) NULL DEFAULT 'pending',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (order_id),
  KEY idx_order_user_id (user_id),
  CONSTRAINT fk_order_user
    FOREIGN KEY (user_id) REFERENCES `user` (user_id)
    ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS orders (
  order_id INT NOT NULL AUTO_INCREMENT,
  user_id INT NULL,
  status VARCHAR(32) NULL DEFAULT 'pending',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (order_id),
  KEY idx_orders_user_id (user_id),
  CONSTRAINT fk_orders_user
    FOREIGN KEY (user_id) REFERENCES `user` (user_id)
    ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS order_items (
  order_item_id INT NOT NULL AUTO_INCREMENT,
  order_id INT NOT NULL,
  product_id VARCHAR(64) NOT NULL,
  qty INT NOT NULL DEFAULT 1,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (order_item_id),
  KEY idx_order_items_order_id (order_id),
  KEY idx_order_items_product_id (product_id),
  CONSTRAINT fk_order_items_order
    FOREIGN KEY (order_id) REFERENCES orders (order_id)
    ON DELETE CASCADE,
  CONSTRAINT fk_order_items_product
    FOREIGN KEY (product_id) REFERENCES products (product_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS face (
  face_id INT NOT NULL AUTO_INCREMENT,
  user_id INT NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (face_id),
  KEY idx_face_user_id (user_id),
  CONSTRAINT fk_face_user
    FOREIGN KEY (user_id) REFERENCES `user` (user_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS facetype (
  facetype_id INT NOT NULL AUTO_INCREMENT,
  face_id INT NOT NULL,
  facetype_name VARCHAR(100) NOT NULL,
  PRIMARY KEY (facetype_id),
  KEY idx_facetype_face_id (face_id),
  CONSTRAINT fk_facetype_face
    FOREIGN KEY (face_id) REFERENCES face (face_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS eyeshape (
  eyeshape_id INT NOT NULL AUTO_INCREMENT,
  face_id INT NOT NULL,
  shape_name VARCHAR(100) NOT NULL,
  PRIMARY KEY (eyeshape_id),
  KEY idx_eyeshape_face_id (face_id),
  CONSTRAINT fk_eyeshape_face
    FOREIGN KEY (face_id) REFERENCES face (face_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS hairstyle (
  hairstyle_id INT NOT NULL AUTO_INCREMENT,
  face_id INT NOT NULL,
  hairstyle_name VARCHAR(150) NOT NULL,
  PRIMARY KEY (hairstyle_id),
  KEY idx_hairstyle_face_id (face_id),
  CONSTRAINT fk_hairstyle_face
    FOREIGN KEY (face_id) REFERENCES face (face_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS face_product (
  face_product_id INT NOT NULL AUTO_INCREMENT,
  face_id INT NOT NULL,
  product_id VARCHAR(64) NOT NULL,
  PRIMARY KEY (face_product_id),
  KEY idx_face_product_face_id (face_id),
  KEY idx_face_product_product_id (product_id),
  CONSTRAINT fk_face_product_face
    FOREIGN KEY (face_id) REFERENCES face (face_id)
    ON DELETE CASCADE,
  CONSTRAINT fk_face_product_product
    FOREIGN KEY (product_id) REFERENCES products (product_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS haircolor (
  haircolor_id INT NOT NULL AUTO_INCREMENT,
  haircolor_name VARCHAR(100) NOT NULL,
  PRIMARY KEY (haircolor_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS liptone (
  liptone_id INT NOT NULL AUTO_INCREMENT,
  liptone_name VARCHAR(100) NOT NULL,
  PRIMARY KEY (liptone_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS skintone (
  skintone_id INT NOT NULL AUTO_INCREMENT,
  skintone_name VARCHAR(100) NOT NULL,
  PRIMARY KEY (skintone_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS productColor (
  productColor_id INT NOT NULL AUTO_INCREMENT,
  product_id VARCHAR(64) NOT NULL,
  color_name VARCHAR(100) NOT NULL,
  PRIMARY KEY (productColor_id),
  KEY idx_productColor_product_id (product_id),
  CONSTRAINT fk_productColor_product
    FOREIGN KEY (product_id) REFERENCES products (product_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS productType (
  productType_id INT NOT NULL AUTO_INCREMENT,
  type_name VARCHAR(100) NOT NULL,
  PRIMARY KEY (productType_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS status (
  status_id INT NOT NULL AUTO_INCREMENT,
  status_name VARCHAR(100) NOT NULL,
  PRIMARY KEY (status_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS eyebrow (
  eyebrow_id INT NOT NULL AUTO_INCREMENT,
  name VARCHAR(150) NOT NULL,
  price DECIMAL(10,2) NULL DEFAULT 0.00,
  image TEXT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  deleted_at DATETIME NULL,
  PRIMARY KEY (eyebrow_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
