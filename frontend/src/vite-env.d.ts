/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Backend origin for the web build (e.g. https://api.enjoi.dev). */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
