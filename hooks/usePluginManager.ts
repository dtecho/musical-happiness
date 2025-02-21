
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useState, useEffect } from 'react';

export interface Plugin {
  id: string;
  name: string;
  version: string;
  enabled: boolean;
  initialize?: () => Promise<void>;
  cleanup?: () => Promise<void>;
}

export interface PluginState {
  enabled: boolean;
  settings: Record<string, any>;
}

export function usePluginManager() {
  const [pluginStates, setPluginStates] = useState<Record<string, PluginState>>({});
  const [plugins, setPlugins] = useState<Plugin[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadPlugins();
  }, []);

  const loadPlugins = async () => {
    try {
      const storedPlugins = await AsyncStorage.getItem('installed_plugins');
      if (storedPlugins) {
        setPlugins(JSON.parse(storedPlugins));
      }
      setLoading(false);
    } catch (error) {
      console.error('Failed to load plugins:', error);
      setLoading(false);
    }
  };

  const togglePlugin = async (pluginId: string): Promise<boolean> => {
    try {
      const plugin = plugins.find(p => p.id === pluginId);
      if (!plugin) return false;
      
      const updatedPlugins = plugins.map(p => 
        p.id === pluginId ? { ...p, enabled: !p.enabled } : p
      );
      
      await AsyncStorage.setItem('installed_plugins', JSON.stringify(updatedPlugins));
      setPlugins(updatedPlugins);
      
      const newState = {
        ...pluginStates,
        [pluginId]: {
          ...pluginStates[pluginId],
          enabled: !plugin.enabled
        }
      };
      setPluginStates(newState);
      await AsyncStorage.setItem('plugin_states', JSON.stringify(newState));
      
      return true;
    } catch (error) {
      console.error('Failed to toggle plugin:', error);
      return false;
    }
  };

  const installPlugin = async (plugin: Plugin) => {
    try {
      const updatedPlugins = [...plugins, plugin];
      await AsyncStorage.setItem('installed_plugins', JSON.stringify(updatedPlugins));
      if (plugin.initialize) {
        await plugin.initialize();
      }
      setPlugins(updatedPlugins);
      return true;
    } catch (error) {
      console.error('Failed to install plugin:', error);
      return false;
    }
  };

  const uninstallPlugin = async (pluginId: string) => {
    try {
      const plugin = plugins.find(p => p.id === pluginId);
      if (plugin?.cleanup) {
        await plugin.cleanup();
      }
      const updatedPlugins = plugins.filter(p => p.id !== pluginId);
      await AsyncStorage.setItem('installed_plugins', JSON.stringify(updatedPlugins));
      setPlugins(updatedPlugins);
      return true;
    } catch (error) {
      console.error('Failed to uninstall plugin:', error);
      return false;
    }
  };

  const updatePluginSettings = async (pluginId: string, settings: Record<string, any>): Promise<void> => {
    const newState = {
      ...pluginStates,
      [pluginId]: {
        ...pluginStates[pluginId],
        settings: {
          ...pluginStates[pluginId]?.settings,
          ...settings
        }
      }
    };
    setPluginStates(newState);
    await AsyncStorage.setItem('plugin_states', JSON.stringify(newState));
  };

  return {
    plugins,
    loading,
    pluginStates,
    installPlugin,
    uninstallPlugin,
    togglePlugin,
    updatePluginSettings
  };
}
