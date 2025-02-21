import {
  Font,
  MD3Type,
  MD3Typescale,
} from 'react-native-paper/lib/typescript/types';

type AppFontKey =
  | 'Brand'
  | 'BrandBold'
  | 'BrandSemiBold'
  | 'OpenSans'
  | 'OnestRegular'
  | 'OnestMedium'
  | 'OnestSemiBold';

/**
 * Getting the fontFamily right is tricky - on iOS you need to use the
 * PostScript name (on MacOS you can get this info from the Identifiers section
 * of FontBook's font info pane - use Cmd+I to toggle), but on Android you need
 * to use the name of the ttf file minus the extension. Ideally these are one
 * and the same, but if that's not possible then use Platform.select to return
 * the correct fontFamily for each platform.
 *
 * NOTE: make sure to run `npx react-native-asset` after any changes to this
 * file or the fonts themselves.
 *
 * Also, variable fonts are not currently supported by react-native. Revisit
 * using them if any of the open PRs land:
 * https://github.com/search?q=repo%3Afacebook%2Freact-native+font+variation+settings&type=pullrequests
 */
export const AppFonts: Record<AppFontKey, Required<Font>> = {
  Brand: {
    fontFamily: 'Outfit-Regular',
    fontWeight: '500',
    fontStyle: 'normal',
  },
  BrandBold: {
    fontFamily: 'Outfit-Bold',
    fontWeight: 'bold',
    fontStyle: 'normal',
  },
  BrandSemiBold: {
    fontFamily: 'Outfit-Medium',
    fontWeight: '600',
    fontStyle: 'normal',
  },
  OpenSans: {
    // this font family has not been working on Android (with Roboto getting
    // used instead) and should actually be 'OpenSans-Regular' - since it's
    // going to be deprecated after v2 themes are launched we'll leave it as-is
    fontFamily: 'Open Sans',
    fontWeight: '400',
    fontStyle: 'normal',
  },
  // Onest fonts are downloaded from https://fonts.google.com/specimen/Onest
  OnestRegular: {
    fontFamily: 'Onest-Regular',
    fontWeight: '400',
    fontStyle: 'normal',
  },
  OnestMedium: {
    fontFamily: 'Onest-Medium',
    fontWeight: '500',
    fontStyle: 'normal',
  },
  OnestSemiBold: {
    fontFamily: 'Onest-SemiBold',
    fontWeight: '600',
    fontStyle: 'normal',
  },
};

export const fontConfig: MD3Typescale = {
  default: {
    fontFamily: AppFonts.OpenSans.fontFamily,
    fontWeight: '400',
    letterSpacing: 0,
  },
  displaySmall: {
    fontFamily: AppFonts.OpenSans.fontFamily,
    fontSize: 36,
    fontWeight: '400',
    letterSpacing: 0,
    lineHeight: 44,
  },
  displayMedium: {
    fontFamily: AppFonts.OpenSans.fontFamily,
    fontSize: 45,
    fontWeight: '400',
    letterSpacing: 0,
    lineHeight: 52,
  },
  displayLarge: {
    fontFamily: AppFonts.OpenSans.fontFamily,
    fontSize: 57,
    fontWeight: '400',
    letterSpacing: 0,
    lineHeight: 64,
  },
  headlineSmall: {
    fontFamily: AppFonts.OpenSans.fontFamily,
    fontSize: 24,
    fontWeight: '400',
    letterSpacing: 0,
    lineHeight: 32,
  },
  headlineMedium: {
    fontFamily: AppFonts.OpenSans.fontFamily,
    fontSize: 28,
    fontWeight: '400',
    letterSpacing: 0,
    lineHeight: 36,
  },
  headlineLarge: {
    fontFamily: AppFonts.OpenSans.fontFamily,
    fontSize: 32,
    fontWeight: '400',
    letterSpacing: 0,
    lineHeight: 40,
  },
  titleSmall: {
    fontFamily: AppFonts.OpenSans.fontFamily,
    fontSize: 14,
    fontWeight: '500',
    letterSpacing: 0.1,
    lineHeight: 20,
  },
  titleMedium: {
    fontFamily: AppFonts.OpenSans.fontFamily,
    fontSize: 16,
    fontWeight: '500',
    letterSpacing: 0.15,
    lineHeight: 24,
  },
  titleLarge: {
    fontFamily: AppFonts.OpenSans.fontFamily,
    fontSize: 22,
    fontWeight: '400',
    letterSpacing: 0,
    lineHeight: 28,
  },
  labelSmall: {
    fontFamily: AppFonts.OpenSans.fontFamily,
    fontSize: 11,
    fontWeight: '500',
    letterSpacing: 0.5,
    lineHeight: 16,
  },
  labelMedium: {
    fontFamily: AppFonts.OpenSans.fontFamily,
    fontSize: 12,
    fontWeight: '500',
    letterSpacing: 0.5,
    lineHeight: 16,
  },
  labelLarge: {
    fontFamily: AppFonts.OpenSans.fontFamily,
    fontSize: 14,
    fontWeight: '500',
    letterSpacing: 0.1,
    lineHeight: 20,
  },
  bodySmall: {
    fontFamily: AppFonts.OpenSans.fontFamily,
    fontSize: 12,
    fontWeight: '400',
    letterSpacing: 0.4,
    lineHeight: 16,
  },
  bodyMedium: {
    fontFamily: AppFonts.OpenSans.fontFamily,
    fontSize: 14,
    fontWeight: '400',
    letterSpacing: 0.25,
    lineHeight: 20,
  },
  bodyLarge: {
    fontFamily: AppFonts.OpenSans.fontFamily,
    fontSize: 16,
    fontWeight: '400',
    letterSpacing: 0.15,
    lineHeight: 24,
  },
};

const regularType: Omit<MD3Type, 'lineHeight' | 'fontSize'> = {
  ...AppFonts.OnestRegular,
  letterSpacing: 0,
};

const mediumType: Omit<MD3Type, 'lineHeight' | 'fontSize'> = {
  ...AppFonts.OnestMedium,
  letterSpacing: 0,
};

const semiboldType: Omit<MD3Type, 'lineHeight' | 'fontSize'> = {
  ...AppFonts.OnestSemiBold,
  letterSpacing: 0,
};

export enum CharacterTypescaleKey {
  display = 'display',
  textXlMedium = 'textXlMedium',
  textLg = 'textLg',
  textLgMedium = 'textLgMedium',
  textMd = 'textMd',
  textMdMedium = 'textMdMedium',
  textMdSemiBold = 'textMdSemiBold',
  textSm = 'textSm',
  textSmMedium = 'textSmMedium',
  textSmSemiBold = 'textSmSemiBold',
}

export type CharacterTypescale = {
  [key in CharacterTypescaleKey]: MD3Type;
};

const fontV2CharacterConfig: CharacterTypescale = {
  display: {
    ...mediumType,
    fontSize: 24,
    lineHeight: 32,
  },
  textXlMedium: {
    ...mediumType,
    fontSize: 20,
    lineHeight: 28,
  },
  textLg: {
    ...regularType,
    fontSize: 16,
    lineHeight: 24,
  },
  textLgMedium: {
    ...mediumType,
    fontSize: 16,
    lineHeight: 24,
  },
  textMd: {
    ...regularType,
    fontSize: 14,
    lineHeight: 20,
  },
  textMdMedium: {
    ...mediumType,
    fontSize: 14,
    lineHeight: 20,
  },
  textMdSemiBold: {
    ...semiboldType,
    fontSize: 14,
    lineHeight: 20,
  },
  textSm: {
    ...regularType,
    fontSize: 12,
    lineHeight: 16,
  },
  textSmMedium: {
    ...mediumType,
    fontSize: 12,
    lineHeight: 16,
  },
  textSmSemiBold: {
    ...semiboldType,
    fontSize: 12,
    lineHeight: 16,
  },
};

const fontV2BaseConfig: MD3Typescale = {
  default: {
    ...regularType,
  },
  // TODO(suman): deprecate after removing any usages
  displaySmall: {
    ...regularType,
    fontSize: 36,
    lineHeight: 32,
  },
  // TODO(suman): deprecate after removing any usages
  displayMedium: {
    ...regularType,
    fontSize: 45,
    lineHeight: 32,
  },
  // TODO(suman): deprecate after removing any usages
  displayLarge: {
    ...regularType,
    fontSize: 57,
    lineHeight: 32,
  },
  headlineSmall: fontV2CharacterConfig.display,
  headlineMedium: fontV2CharacterConfig.display,
  // TODO(suman): deprecate after removing any usages
  headlineLarge: {
    ...mediumType,
    fontSize: 32,
    lineHeight: 32,
  },
  titleSmall: fontV2CharacterConfig.textMdMedium,
  titleMedium: fontV2CharacterConfig.textLgMedium,
  titleLarge: fontV2CharacterConfig.textXlMedium,
  labelSmall: fontV2CharacterConfig.textSm,
  labelMedium: fontV2CharacterConfig.textSmSemiBold,
  labelLarge: fontV2CharacterConfig.textMdMedium,
  bodySmall: fontV2CharacterConfig.textSm,
  bodyMedium: fontV2CharacterConfig.textMd,
  bodyLarge: fontV2CharacterConfig.textLg,
};

export const fontV2Config: MD3Typescale & CharacterTypescale = {
  ...fontV2BaseConfig,
  ...fontV2CharacterConfig,
};
