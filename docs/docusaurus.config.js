// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'sophus2',
  tagline: 'A collection of c++ types for 2d and 3d geometric problems',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://strasdat.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: 'Sophus/latest/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'strasdat', // Usually your GitHub org/user name.
  projectName: 'Sophus', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      navbar: {
        title: 'sophus2',
        logo: {
          alt: '>_',
          src: 'img/farm-ng_favicon.png',
        },
        items: [
          {
            type: 'doc',
            docId: 'intro',
            position: 'left',
            label: 'Book',
          },
          {
            href: 'https://farm-ng.github.io/docs/namespacesophus.html',
            position: 'left',
            label: 'C++ API',
          },
          {
            href: 'https://github.com/strasdat/sophus/tree/sophus2',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Supported by farm-ng inc.',
            items: [
              {
                label: 'website',
                href: 'https://farm-ng.com/',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Github Issues',
                href: 'https://github.com/strasdat/Sophus/issues',
              },
              {
                label: 'Github Discussions',
                href: 'https://github.com/strasdat/Sophus/discussions',
              },
            ],
          },
          {
            title: 'farm-ng and sister projects',
            items: [
              {
                label: 'farm-ng-core',
                href: 'https://farm-ng.github.io/farm-ng-core/',
              },
              {
                label: 'Sophus',
                href: 'https://strasdat.github.io/Sophus/latest/',
              },
              {
                label: 'Pangolin',
                href: 'https://github.com/stevenlovegrove/Pangolin',
              },
              {
                label: 'Amiga Development Kit',
                href: 'https://amiga.farm-ng.com/docs/getting-started',
              },
            ],
          },
        ],
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
};

module.exports = config;
