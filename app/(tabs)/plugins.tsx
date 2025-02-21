import React from 'react';
import { View, FlatList, StyleSheet, ActivityIndicator, Pressable } from 'react-native';
import { usePluginManager } from '../../hooks/usePluginManager';
import { getAvailablePlugins } from '../../plugins/registry';
import ThemedText from '../../components/ThemedText';
import ThemedView from '../../components/ThemedView';
import { PluginSettings } from '../../components/ui/PluginSettings';

export default function PluginsScreen() {
  const { plugins, loading, installPlugin, uninstallPlugin, togglePlugin } = usePluginManager();
  const availablePlugins = getAvailablePlugins();

  const handleInstall = async (plugin) => {
    await installPlugin(plugin);
  };

  if (loading) {
    return (
      <ThemedView style={styles.centered}>
        <ActivityIndicator size="large" />
      </ThemedView>
    );
  }

  return (
    <ThemedView style={styles.container}>
      <FlatList
        data={availablePlugins}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => {
          const isInstalled = plugins.some(p => p.id === item.id);
          const isEnabled = plugins.find(p => p.id === item.id)?.enabled;

          return (
            <View style={styles.pluginCard}>
              <ThemedText style={styles.pluginName}>{item.name}</ThemedText>
              <ThemedText style={styles.version}>v{item.version}</ThemedText>
              <Pressable
                style={[styles.button, isInstalled ? styles.uninstallButton : styles.installButton]}
                onPress={() => isInstalled ? uninstallPlugin(item.id) : handleInstall(item)}
              >
                <ThemedText>{isInstalled ? 'Uninstall' : 'Install'}</ThemedText>
              </Pressable>
            </View>
          );
        }}
      />
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  pluginCard: {
    padding: 16,
    borderRadius: 8,
    marginBottom: 16,
    backgroundColor: '#2C2C2C',
  },
  pluginName: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  version: {
    opacity: 0.7,
    marginBottom: 12,
  },
  button: {
    padding: 8,
    borderRadius: 4,
    alignItems: 'center',
  },
  installButton: {
    backgroundColor: '#4CAF50',
  },
  uninstallButton: {
    backgroundColor: '#f44336',
  },
});