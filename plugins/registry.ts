
import type { Plugin } from '../hooks/usePluginManager';
import { llamaPlugin } from './llama';
import { RWKVPlugin } from './rwkv';
import { mem0Plugin } from './mem0';

const pluginRegistry = {
  llama: llamaPlugin,
  rwkv: new RWKVPlugin(),
  mem0: mem0Plugin,
};

export type PluginKey = keyof typeof pluginRegistry;
export const getAvailablePlugins = () => Object.values(pluginRegistry);
export default pluginRegistry;
