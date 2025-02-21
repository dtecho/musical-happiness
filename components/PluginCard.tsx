
import React from 'react';
import { View, Switch, StyleSheet } from 'react-native';
import { ThemedText } from './ThemedText';
import { usePluginManager } from '../hooks/usePluginManager';

export const PluginCard = ({ pluginId, name, description }) => {
  const { pluginStates, togglePlugin } = usePluginManager();
  
  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <ThemedText style={styles.title}>{name}</ThemedText>
        <Switch 
          value={pluginStates[pluginId]?.enabled ?? false}
          onValueChange={() => togglePlugin(pluginId)}
        />
      </View>
      <ThemedText style={styles.description}>{description}</ThemedText>
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    padding: 16,
    borderRadius: 8,
    marginVertical: 8,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  description: {
    marginTop: 8,
    opacity: 0.7,
  },
});
