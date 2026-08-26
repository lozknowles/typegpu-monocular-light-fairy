import typegpu from 'eslint-plugin-typegpu';
import { defineConfig } from 'oxlint';

export default defineConfig({
  jsPlugins: ['eslint-plugin-typegpu'],
  rules: {
    ...typegpu.configs.recommended.rules,
  },
  ignorePatterns: ['dist/**', 'node_modules/**', '**/*.tsnotover.ts'],
});
