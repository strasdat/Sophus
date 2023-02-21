import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  //Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: '2D / 3d Building Blocks',
    //Svg: require('@site/static/img/Farm-ng_Logo_Black.svg').default,
    description: (
      <>
        Collections of Lie Groups / Manifolds commonly used for 2d / 3D
        geometric problems.
      </>
    ),
  },
  {
    title: 'Image Classes',
    //Svg: require('@site/static/img/Farm-ng_Logo_Black.svg').default,
    description: (
      <>
        Image, MutImage, DynImage, MutDynImage.
      </>
    ),
  },
  {
    title: 'Camera Models and More',
    //Svg: require('@site/static/img/Farm-ng_Logo_Black.svg').default,
    description: (
      <>
        Collection of camera models (pinhole, brown-conrady aka opencv,
        kannala-brandt and orthographic), IMU mode and more.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
