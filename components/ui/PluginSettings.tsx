
import React, { useState } from 'react';
import { View, TextInput, Switch, StyleSheet } from 'react-native';
import ThemedText from '../ThemedText';
import { usePluginManager } from '../../hooks/usePluginManager';

export const PluginSettings = () => {
  const { plugins, togglePlugin } = usePluginManager();
  const [llamaConfig, setLlamaConfig] = useState({
    modelPath: '',
    contextSize: '2048',
    threads: '4'
  });
  
  const [mem0Config, setMem0Config] = useState({
    apiKey: ''
  });

  return (
    <View style={styles.container}>
      <ThemedText style={styles.title}>Plugin Settings</ThemedText>
      
      {plugins.map((plugin) => (
        <View key={plugin.id} style={styles.pluginContainer}>
          <View style={styles.header}>
            <ThemedText style={styles.pluginName}>{plugin.name}</ThemedText>
            <Switch
              value={plugin.enabled}
              onValueChange={() => togglePlugin(plugin.id)}
            />
          </View>
          
          {plugin.id === 'llama-plugin' && plugin.enabled && (
            <View style={styles.configContainer}>
              <TextInput
                style={styles.input}
                placeholder="Model Path"
                value={llamaConfig.modelPath}
                onChangeText={(text) => setLlamaConfig({...llamaConfig, modelPath: text})}
              />
              <TextInput
                style={styles.input}
                placeholder="Context Size"
                value={llamaConfig.contextSize}
                onChangeText={(text) => setLlamaConfig({...llamaConfig, contextSize: text})}
                keyboardType="numeric"
              />
              <TextInput
                style={styles.input}
                placeholder="Threads"
                value={llamaConfig.threads}
                onChangeText={(text) => setLlamaConfig({...llamaConfig, threads: text})}
                keyboardType="numeric"
              />
            </View>
          )}
          
          {plugin.id === 'mem0-plugin' && plugin.enabled && (
            <View style={styles.configContainer}>
              <TextInput
                style={styles.input}
                placeholder="API Key"
                value={mem0Config.apiKey}
                onChangeText={(text) => setMem0Config({...mem0Config, apiKey: text})}
                secureTextEntry
              />
            </View>
          )}
        </View>
      ))}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  pluginContainer: {
    marginBottom: 16,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  pluginName: {
    fontSize: 18,
    fontWeight: '500',
  },
  configContainer: {
    marginTop: 8,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 4,
    padding: 8,
    marginBottom: 8,
  },
});
