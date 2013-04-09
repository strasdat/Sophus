#include <iostream>

#include <ceres/ceres.h>
#include <sophus/se3.hpp>

#include "ceres_eigen.hpp"
#include "local_parameterization_se3.hpp"

using namespace Sophus;
using namespace Eigen;
using namespace ceres;

struct TestCostFunctor
{
    TestCostFunctor(SE3d T_aw)
        : T_aw(T_aw)
    {
    }
    
    template<typename T>
    bool operator()( const T* const sT_wa, T* sResiduals ) const
    {
        const Eigen::Map<const Sophus::SE3Group<T> > T_wa(sT_wa);
        Eigen::Map<Eigen::Matrix<T,6,1> > residuals(sResiduals);
        
        residuals = (T_aw.cast<T>() * T_wa).log();
        return true;
    }
    
    SE3d T_aw;
};

bool test(const SE3d& T_w_targ, const SE3d& T_w_init)
{
    // Optimisation parameter
    SE3d T_wr = T_w_init;
    
    // Build the problem.
    Problem problem;
    
    // Specify local update rule for our parameter
    problem.AddParameterBlock(T_wr.data(), SE3d::num_parameters, new LocalParameterizationSe3);
    
    // Create and add cost function. Derivatives will be evaluated via
    // automatic differentiation
    CostFunction* cost_function =
        new AutoDiffCostFunction<TestCostFunctor, SE3d::DoF, SE3d::num_parameters>(
            new TestCostFunctor(T_w_targ.inverse())
        );
    problem.AddResidualBlock(cost_function, NULL, T_wr.data());
    
    // Set solver options (precision / method)
    Solver::Options options;
    options.gradient_tolerance = 0.01 * SophusConstants<double>::epsilon();
    options.function_tolerance = 0.01 * SophusConstants<double>::epsilon();
    options.linear_solver_type = ceres::DENSE_QR;
    
    // Solve
    Solver::Summary summary;
    Solve(options, &problem, &summary);    
    cout << summary.BriefReport() << endl;

    // Difference between target and parameter
    const double mse = (T_w_targ.inverse() * T_wr).log().squaredNorm();
    const bool passed = mse < 10. * SophusConstants<double>::epsilon();    
    return passed;
}

int main(int, char**)
{
    typedef SE3Group<double> SE3Type;
    typedef SO3Group<double> SO3Type;
    typedef typename SE3Group<double>::Point Point;    
    
    vector<SE3Type> se3_vec;
    se3_vec.push_back(SE3Type(SO3Type::exp(Point(0.2, 0.5, 0.0)),
                              Point(0,0,0)));
    se3_vec.push_back(SE3Type(SO3Type::exp(Point(0.2, 0.5, -1.0)),
                              Point(10,0,0)));
    se3_vec.push_back(SE3Type(SO3Type::exp(Point(0., 0., 0.)),
                              Point(0,100,5)));
    se3_vec.push_back(SE3Type(SO3Type::exp(Point(0., 0., 0.00001)),
                              Point(0,0,0)));
    se3_vec.push_back(SE3Type(SO3Type::exp(Point(0., 0., 0.00001)),
                              Point(0,-0.00000001,0.0000000001)));
    se3_vec.push_back(SE3Type(SO3Type::exp(Point(0., 0., 0.00001)),
                              Point(0.01,0,0)));
    se3_vec.push_back(SE3Type(SO3Type::exp(Point(M_PI, 0, 0)),
                              Point(4,-5,0)));
    se3_vec.push_back(SE3Type(SO3Type::exp(Point(0.2, 0.5, 0.0)),
                              Point(0,0,0))
                      *SE3Type(SO3Type::exp(Point(M_PI, 0, 0)),
                               Point(0,0,0))
                      *SE3Type(SO3Type::exp(Point(-0.2, -0.5, -0.0)),
                               Point(0,0,0)));
    se3_vec.push_back(SE3Type(SO3Type::exp(Point(0.3, 0.5, 0.1)),
                              Point(2,0,-7))
                      *SE3Type(SO3Type::exp(Point(M_PI, 0, 0)),
                               Point(0,0,0))
                      *SE3Type(SO3Type::exp(Point(-0.3, -0.5, -0.1)),
                               Point(0,6,0)));

    
    for(size_t i=0; i < se3_vec.size(); ++i )
    {
        const bool passed = test( se3_vec[i], se3_vec[(i+3) % se3_vec.size()] );
        if (!passed) {
          cerr << "failed!" << endl << endl;
          exit(-1);
        }
    }
    
    return 0;
}
