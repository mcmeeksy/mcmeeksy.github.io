import React from 'react'; 
import { NavigationContainer, Theme } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { StatusBar } from 'expo-status-bar';
import { Ionicons } from '@expo/vector-icons';
import HomeScreen from './src/screens/HomeScreen';
import WalkScreen from './src/screens/WalkScreen';
import ExploreScreen from './src/screens/ExploreScreen';
import FriendsScreen from './src/screens/FriendsScreen';
import ProfileScreen from './src/screens/ProfileScreen';

export type RootTabParamList = {
  Home: undefined;
  Walk: undefined;
  Explore: undefined;
  Friends: undefined;
  Profile: undefined;
};

const Tab = createBottomTabNavigator<RootTabParamList>();

const theme: Theme = {
  dark: false,
  colors: {
    primary: '#315C4A',
    background: '#F7F5EF',
    card: '#FFFFFF',
    text: '#24312B',
    border: '#E4E1D8',
    notification: '#C96A4A',
  },
  fonts: {
    regular: { fontFamily: 'System', fontWeight: '400' },
    medium: { fontFamily: 'System', fontWeight: '500' },
    bold: { fontFamily: 'System', fontWeight: '700' },
    heavy: { fontFamily: 'System', fontWeight: '800' },
  },
};

const icons: Record<keyof RootTabParamList, keyof typeof Ionicons.glyphMap> = {
  Home: 'paw-outline',
  Walk: 'walk-outline',
  Explore: 'map-outline',
  Friends: 'people-outline',
  Profile: 'person-outline',
};

export default function App() {
  return (
    <NavigationContainer theme={theme}>
      <StatusBar style="dark" />
      <Tab.Navigator
        initialRouteName="Home"
        screenOptions={({ route }) => ({
          headerShown: false,
          tabBarActiveTintColor: '#315C4A',
          tabBarInactiveTintColor: '#8A918C',
          tabBarStyle: {
            height: 74,
            paddingTop: 8,
            paddingBottom: 10,
            backgroundColor: '#FFFFFF',
            borderTopColor: '#E4E1D8',
          },
          tabBarLabelStyle: {
            fontSize: 11,
            fontWeight: '600',
          },
          tabBarIcon: ({ color, size }) => (
            <Ionicons name={icons[route.name]} size={size} color={color} />
          ),
        })}
      >
        <Tab.Screen name="Home" component={HomeScreen} options={{ title: 'Home' }} />
        <Tab.Screen name="Walk" component={WalkScreen} options={{ title: 'Walk' }} />
        <Tab.Screen name="Explore" component={ExploreScreen} options={{ title: 'Explore' }} />
        <Tab.Screen name="Friends" component={FriendsScreen} options={{ title: 'Friends' }} />
        <Tab.Screen name="Profile" component={ProfileScreen} options={{ title: 'Profile' }} />
      </Tab.Navigator>
    </NavigationContainer>
  );
}
