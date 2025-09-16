#pragma once


#include "ceres/ceres.h"

#include <sophus/ceres_manifold.hpp>
#include <sophus/spline.hpp>

namespace Sophus {

    template <typename Scalar_,template <typename, int = 0> class LieGroup_>
        struct SplineErrorSupport {
            typedef enum {
                SPLINE_P0=0,
                SPLINE_P1=1,
                SPLINE_P2=2,
                SPLINE_P3=3,
                SPLINE_Q0=4,
                SPLINE_Q1=5,
                SPLINE_Q2=6,
                SPLINE_Q3=7,
            } SplineParameterId;

            template <typename T>
                using LieGroup = LieGroup_<T>;
            using LieGroupd = LieGroup<Scalar_>;
            using Splined = Sophus::BasisSpline<LieGroupd>;
            static int constexpr num_parameters = LieGroupd::num_parameters;

            template <class ErrorFunctor, int num_residuals_> 
                struct SplineErrorWrapper0 {
                    static int constexpr num_residuals = num_residuals_;
                    // using SplineErrorWrapper<ErrorFunctor>::call;
                    SplineErrorWrapper0(Sophus::SegmentCase scase, double u, 
                            std::shared_ptr<ErrorFunctor> soph, const std::vector<unsigned char> & pmap) : 
                        pmap(pmap), segment_case(scase), derivative_order(0), u(u), delta_t(0.0), soph(soph) {
                        }

                    SplineErrorWrapper0(unsigned int derivative_order, Sophus::SegmentCase scase, double u, double delta_t, 
                            std::shared_ptr<ErrorFunctor> soph, const std::vector<unsigned char> & pmap) : 
                        pmap(pmap), segment_case(scase), derivative_order(derivative_order), u(u), delta_t(delta_t), soph(soph) {
                        }

                    void check_map(unsigned char maxval) const {
                        for (auto x : this->pmap) {
                            SOPHUS_ENSURE(x <= maxval, "but %", x);
                        }
                    }

                    template <typename T>
                        bool operator()(const T* P0, const T* P1, const T* P2, 
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2};
                            //alias
                            const std::vector<unsigned char> & pmap = this->pmap;
                            this->check_map(3);
                            return this->template call<T, T>(P[pmap[SPLINE_P0]],
                                    P[pmap[SPLINE_P1]],
                                    P[pmap[SPLINE_P2]],
                                    P[pmap[SPLINE_P3]],residuals);
                        }

                    template <typename T>
                        bool operator()(const T* P0, const T* P1, const T* P2, const T* P3, 
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2,P3};
                            //alias
                            const std::vector<unsigned char> & pmap = this->pmap;
                            this->check_map(4);
                            return this->template call<T, T>(P[pmap[SPLINE_P0]],
                                    P[pmap[SPLINE_P1]],
                                    P[pmap[SPLINE_P2]],
                                    P[pmap[SPLINE_P3]],residuals);
                        }


                    template <typename T2, typename T>
                        bool call(const T2* const P0, const T2* const P1, const T2* const P2, const T2* const P3, 
                                T* residuals) const {

                            using LGT2 = LieGroup_<T2>;
                            using dLGT2 = typename LieGroup_<T2>::Transformation;
                            // Mapper class is only used to facciliate difference between
                            // SO2 (which uses Scalar as tangent vector type) and other groups
                            // (which use Vector<...> as tangent vector type).
                            using Mapper = Mapper<typename LieGroup_<T2>::Tangent>;
                            Sophus::BasisSplineSegment<LGT2> s(this->segment_case,P0,P1,P2,P3);
                            if (derivative_order==0) {
                                LGT2 Tt = s.parent_T_spline(this->u);
                                return this->soph->operator()(Tt.data(),residuals);
                            } else if (derivative_order==1) {
                                dLGT2 Tt = s.Dt_parent_T_spline(this->u,this->delta_t);
                                T2 tangent_data[LGT2::DoF];
                                typename Mapper::Map v = Mapper::map(tangent_data);
                                v = LGT2::vee(Tt);
                                return this->soph->operator()(tangent_data,residuals);
                            } else if (derivative_order==2) {
                                dLGT2 Tt = s.Dt2_parent_T_spline(this->u,this->delta_t);
                                T2 tangent_data[LGT2::DoF];
                                typename Mapper::Map v = Mapper::map(tangent_data);
                                v = LGT2::vee(Tt);
                                return this->soph->operator()(tangent_data,residuals);
                            } else {
                                assert(derivative_order < 3); 
                            }
                            return false;
                        }

                    std::vector<unsigned char> pmap;
                    Sophus::SegmentCase segment_case;
                    unsigned int derivative_order;
                    double u, delta_t;
                    std::shared_ptr<ErrorFunctor> soph;

                };

            template <class ErrorFunctor,int num_residuals_> 
                struct SplineError2FunctionsWrapper0 {
                    static int constexpr num_residuals = num_residuals_;
                    SplineError2FunctionsWrapper0(Sophus::SegmentCase scase1, double u1, Sophus::SegmentCase scase2, double u2, 
                            std::shared_ptr<ErrorFunctor> soph,
                            const std::vector<unsigned char> & pmap) : 
                        pmap(pmap),segment_case1(scase1),segment_case2(scase2), 
                        derivative_order(0), u1(u1), u2(u2), delta_t1(0.0), delta_t2(0.0), soph(soph) {
                        }

                    SplineError2FunctionsWrapper0(unsigned int derivative_order, 
                            Sophus::SegmentCase scase1, double u1, double delta_t1, 
                            Sophus::SegmentCase scase2, double u2, double delta_t2, 
                            double delta_t, 
                            std::shared_ptr<ErrorFunctor> soph,
                            const std::vector<unsigned char> & pmap) : 
                        pmap(pmap), segment_case1(scase1), segment_case2(scase2), 
                        derivative_order(derivative_order), u1(u1), u2(u2), delta_t1(delta_t1), delta_t2(delta_t2), soph(soph) {
                        }

                    void check_map(unsigned char maxval) const {
                        for (auto x : pmap) {
                            SOPHUS_ENSURE(x <= maxval, "but %", x);
                        }
                    }

                    template <typename T>
                        bool operator()(const T* P0, const T* P1, const T* P2, const T* P3, const T* P4, const T* P5,
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2,P3,P4,P5};
                            this->check_map(6);
                            //alias
                            const std::vector<unsigned char> & pmap = this->pmap;
                            return this->template call<T, T, T>(P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],
                                    P[pmap[SPLINE_Q0]],P[pmap[SPLINE_Q1]],P[pmap[SPLINE_Q2]],P[pmap[SPLINE_Q3]],residuals);
                        }

                    template <typename T>
                        bool operator()(const T* P0, const T* P1, const T* P2, const T* P3, const T* P4, const T* P5, const T* P6,
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2,P3,P4,P5,P6};
                            this->check_map(7);
                            //alias
                            const std::vector<unsigned char> & pmap = this->pmap;
                            return this->template call<T, T, T>(P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],
                                    P[pmap[SPLINE_Q0]],P[pmap[SPLINE_Q1]],P[pmap[SPLINE_Q2]],P[pmap[SPLINE_Q3]],residuals);
                        }

                    template <typename T>
                        bool operator()(const T* P0, const T* P1, const T* P2, const T* P3, const T* P4, const T* P5, const T* P6, const T* P7,
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2,P3,P4,P5,P6,P7};
                            this->check_map(8);
                            //alias
                            const std::vector<unsigned char> & pmap = this->pmap;
                            return this->template call<T, T, T>(P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],
                                    P[pmap[SPLINE_Q0]],P[pmap[SPLINE_Q1]],P[pmap[SPLINE_Q2]],P[pmap[SPLINE_Q3]],residuals);
                        }


                    template <typename T1, typename T2, typename T>
                        bool call(const T1* const P0, const T1* const P1, const T1* const P2, const T1* const P3, 
                                const T2* const Q0, const T2* const Q1, const T2* const Q2, const T2* const Q3, 
                                T* residuals) const {

                            using LGT1 = LieGroup_<T1>;
                            using dLGT1 = typename LieGroup_<T1>::Transformation;
                            // Mapper class is only used to facciliate difference between
                            // SO2 (which uses Scalar as tangent vector type) and other groups
                            // (which use Vector<...> as tangent vector type).
                            using Mapper1 = Mapper<typename LieGroup_<T1>::Tangent>;
                            using LGT2 = LieGroup_<T2>;
                            using dLGT2 = typename LieGroup_<T2>::Transformation;
                            using Mapper2 = Mapper<typename LieGroup_<T2>::Tangent>;
                            Sophus::BasisSplineSegment<LGT1> s1(this->segment_case1,P0,P1,P2,P3);
                            Sophus::BasisSplineSegment<LGT2> s2(this->segment_case2,Q0,Q1,Q2,Q3);
                            if (derivative_order==0) {
                                LGT1 t1 = s1.parent_T_spline(this->u1);
                                LGT2 t2 = s2.parent_T_spline(this->u2);
                                return this->soph->operator()(t1.data(),t2.data(),residuals);
                            } else if (derivative_order==1) {
                                dLGT1 t1 = s1.Dt_parent_T_spline(this->u1,this->delta_t1);
                                dLGT2 t2 = s2.Dt_parent_T_spline(this->u2,this->delta_t2);
                                T1 tangent_data1[LGT1::DoF];
                                typename Mapper1::Map v1 = Mapper1::map(tangent_data1);
                                v1 = LGT2::vee(t1);
                                T2 tangent_data2[LGT2::DoF];
                                typename Mapper2::Map v2 = Mapper2::map(tangent_data2);
                                return this->soph->operator()(tangent_data1,tangent_data2,residuals);
                                v2 = LGT2::vee(t2);
                            } else if (derivative_order==2) {
                                dLGT1 t1 = s1.Dt2_parent_T_spline(this->u1,this->delta_t1);
                                dLGT2 t2 = s2.Dt2_parent_T_spline(this->u2,this->delta_t2);
                                T1 tangent_data1[LGT1::DoF];
                                typename Mapper1::Map v1 = Mapper1::map(tangent_data1);
                                v1 = LGT2::vee(t1);
                                T2 tangent_data2[LGT2::DoF];
                                typename Mapper2::Map v2 = Mapper2::map(tangent_data2);
                                v2 = LGT2::vee(t2);
                                return this->soph->operator()(tangent_data1,tangent_data2,residuals);
                            } else {
                                assert(derivative_order < 3); 
                            }
                            return false;
                        }

                    std::vector<unsigned char> pmap;
                    Sophus::SegmentCase segment_case1, segment_case2;
                    unsigned int derivative_order;
                    double u1, u2, delta_t1, delta_t2;
                    std::shared_ptr<ErrorFunctor> soph;

                };

            template <class ErrorFunctor,int num_residuals_> 
                struct SplineError2PointsWrapper0 {
                    static int constexpr num_residuals = num_residuals_;
                    // using SplineErrorWrapper<ErrorFunctor>::call;
                    SplineError2PointsWrapper0(Sophus::SegmentCase scase1, double u1, Sophus::SegmentCase scase2, double u2, 
                            std::shared_ptr<ErrorFunctor> soph,
                            const std::vector<unsigned char> & pmap) : 
                        pmap(pmap),segment_case1(scase1),segment_case2(scase2), 
                        derivative_order(0), u1(u1), u2(u2), delta_t(0.0), soph(soph) {
                        }

                    SplineError2PointsWrapper0(unsigned int derivative_order, 
                            Sophus::SegmentCase scase1, double u1, 
                            Sophus::SegmentCase scase2, double u2, 
                            double delta_t, 
                            std::shared_ptr<ErrorFunctor> soph,
                            const std::vector<unsigned char> & pmap) : 
                        pmap(pmap), segment_case1(scase1), segment_case2(scase2), 
                        derivative_order(derivative_order), u1(u1), u2(u2), delta_t(delta_t), soph(soph) {
                        }

                    void check_map(unsigned char maxval) const {
                        for (auto x : pmap) {
                            SOPHUS_ENSURE(x <= maxval, "but %", x);
                        }
                    }

                    template <typename T>
                        bool operator()(const T* P0, const T* P1, const T* P2, 
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2};
                            this->check_map(3);
                            //alias
                            const std::vector<unsigned char> & pmap = this->pmap;
                            return this->template call<T, T, T>(P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],
                                    P[pmap[SPLINE_Q0]],P[pmap[SPLINE_Q1]],P[pmap[SPLINE_Q2]],P[pmap[SPLINE_Q3]],residuals);
                        }

                    template <typename T>
                        bool operator()(const T* P0, const T* P1, const T* P2, const T* P3,
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2,P3};
                            this->check_map(4);
                            //alias
                            const std::vector<unsigned char> & pmap = this->pmap;
                            return this->template call<T, T, T>(P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],
                                    P[pmap[SPLINE_Q0]],P[pmap[SPLINE_Q1]],P[pmap[SPLINE_Q2]],P[pmap[SPLINE_Q3]],residuals);
                        }

                    template <typename T>
                        bool operator()(const T* P0, const T* P1, const T* P2, const T* P3, const T* P4,
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2,P3,P4};
                            this->check_map(5);
                            //alias
                            const std::vector<unsigned char> & pmap = this->pmap;
                            return this->template call<T, T, T>(P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],
                                    P[pmap[SPLINE_Q0]],P[pmap[SPLINE_Q1]],P[pmap[SPLINE_Q2]],P[pmap[SPLINE_Q3]],residuals);
                        }

                    template <typename T>
                        bool operator()(const T* P0, const T* P1, const T* P2, const T* P3, const T* P4, const T* P5,
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2,P3,P4,P5};
                            this->check_map(6);
                            //alias
                            const std::vector<unsigned char> & pmap = this->pmap;
                            return this->template call<T, T, T>(P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],
                                    P[pmap[SPLINE_Q0]],P[pmap[SPLINE_Q1]],P[pmap[SPLINE_Q2]],P[pmap[SPLINE_Q3]],residuals);
                        }

                    template <typename T>
                        bool operator()(const T* P0, const T* P1, const T* P2, const T* P3, const T* P4, const T* P5, const T* P6,
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2,P3,P4,P5,P6};
                            this->check_map(7);
                            //alias
                            const std::vector<unsigned char> & pmap = this->pmap;
                            return this->template call<T, T, T>(P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],
                                    P[pmap[SPLINE_Q0]],P[pmap[SPLINE_Q1]],P[pmap[SPLINE_Q2]],P[pmap[SPLINE_Q3]],residuals);
                        }

                    template <typename T>
                        bool operator()(const T* P0, const T* P1, const T* P2, const T* P3, const T* P4, const T* P5, const T* P6, const T* P7,
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2,P3,P4,P5,P6,P7};
                            this->check_map(8);
                            //alias
                            const std::vector<unsigned char> & pmap = this->pmap;
                            return this->template call<T, T, T>(P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],
                                    P[pmap[SPLINE_Q0]],P[pmap[SPLINE_Q1]],P[pmap[SPLINE_Q2]],P[pmap[SPLINE_Q3]],residuals);
                        }


                    template <typename T1, typename T2, typename T>
                        bool call(const T1* const P0, const T1* const P1, const T1* const P2, const T1* const P3, 
                                const T2* const Q0, const T2* const Q1, const T2* const Q2, const T2* const Q3, 
                                T* residuals) const {

                            using LGT1 = LieGroup_<T1>;
                            using dLGT1 = typename LieGroup_<T1>::Transformation;
                            // Mapper class is only used to facciliate difference between
                            // SO2 (which uses Scalar as tangent vector type) and other groups
                            // (which use Vector<...> as tangent vector type).
                            using Mapper1 = Mapper<typename LieGroup_<T1>::Tangent>;
                            using LGT2 = LieGroup_<T2>;
                            using dLGT2 = typename LieGroup_<T2>::Transformation;
                            using Mapper2 = Mapper<typename LieGroup_<T2>::Tangent>;
                            Sophus::BasisSplineSegment<LGT1> s1(this->segment_case1,P0,P1,P2,P3);
                            Sophus::BasisSplineSegment<LGT2> s2(this->segment_case2,Q0,Q1,Q2,Q3);
                            if (derivative_order==0) {
                                LGT1 t1 = s1.parent_T_spline(this->u1);
                                LGT2 t2 = s2.parent_T_spline(this->u2);
                                return this->soph->operator()(t1.data(),t2.data(),residuals);
                            } else if (derivative_order==1) {
                                dLGT1 t1 = s1.Dt_parent_T_spline(this->u1,this->delta_t);
                                dLGT2 t2 = s2.Dt_parent_T_spline(this->u2,this->delta_t);
                                T1 tangent_data1[LGT1::DoF];
                                typename Mapper1::Map v1 = Mapper1::map(tangent_data1);
                                v1 = LGT2::vee(t1);
                                T2 tangent_data2[LGT2::DoF];
                                typename Mapper2::Map v2 = Mapper2::map(tangent_data2);
                                return this->soph->operator()(tangent_data1,tangent_data2,residuals);
                                v2 = LGT2::vee(t2);
                            } else if (derivative_order==2) {
                                dLGT1 t1 = s1.Dt2_parent_T_spline(this->u1,this->delta_t);
                                dLGT2 t2 = s2.Dt2_parent_T_spline(this->u2,this->delta_t);
                                T1 tangent_data1[LGT1::DoF];
                                typename Mapper1::Map v1 = Mapper1::map(tangent_data1);
                                v1 = LGT2::vee(t1);
                                T2 tangent_data2[LGT2::DoF];
                                typename Mapper2::Map v2 = Mapper2::map(tangent_data2);
                                v2 = LGT2::vee(t2);
                                return this->soph->operator()(tangent_data1,tangent_data2,residuals);
                            } else {
                                assert(derivative_order < 3); 
                            }
                            return false;
                        }

                    std::vector<unsigned char> pmap;
                    Sophus::SegmentCase segment_case1, segment_case2;
                    unsigned int derivative_order;
                    double u1, u2, delta_t;
                    std::shared_ptr<ErrorFunctor> soph;

                };

            template <class ErrorFunctor,int num_residuals_> 
                struct SplineErrorWrapper1 : public SplineErrorWrapper0<ErrorFunctor,num_residuals_> {
                    SplineErrorWrapper1(Sophus::SegmentCase scase, double u, std::shared_ptr<ErrorFunctor> soph, const std::vector<unsigned char> & pmap) : 
                        SplineErrorWrapper0<ErrorFunctor,num_residuals_>(scase,u,soph,pmap) {}

                    SplineErrorWrapper1(unsigned int derivative_order, Sophus::SegmentCase scase, double u, double delta_t, std::shared_ptr<ErrorFunctor> soph, const std::vector<unsigned char> & pmap) : 
                        SplineErrorWrapper0<ErrorFunctor,num_residuals_>(derivative_order,scase,u,delta_t,soph,pmap) {}


                    template <typename T>
                        bool operator()(const T* const C0,
                                const T* P0, const T* P1, const T* P2, 
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2};
                            const std::vector<unsigned char> & pmap = this->pmap;
                            this->check_map(3);
                            return this->template call<T, T, T>(C0, P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],
                                    P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],residuals);
                        }

                    template <typename T>
                        bool operator()(const T* const C0,
                                const T* P0, const T* P1, const T* P2, const T* P3,
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2,P3};
                            const std::vector<unsigned char> & pmap = this->pmap;
                            this->check_map(4);
                            return this->template call<T, T, T>(C0, P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],
                                    P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],residuals);
                        }


                    template <typename T1, typename T2, typename T>
                        bool call(const T1* const C0,
                                const T2* const P0, const T2* const P1, const T2* const P2, const T2* const P3, 
                                T* residuals) const {

                            using LGT2 = LieGroup_<T2>;
                            using dLGT2 = typename LieGroup_<T2>::Transformation;
                            // Mapper class is only used to facciliate difference between
                            // SO2 (which uses Scalar as tangent vector type) and other groups
                            // (which use Vector<...> as tangent vector type).
                            using Mapper2 = Mapper<typename LieGroup_<T2>::Tangent>;
                            Sophus::BasisSplineSegment<LGT2> s(this->segment_case,P0,P1,P2,P3);
                            if (this->derivative_order==0) {
                                LGT2 Tt = s.parent_T_spline(this->u);
                                return this->soph->operator()(C0,Tt.data(),residuals);
                            } else if (this->derivative_order==1) {
                                dLGT2 Tt = s.Dt_parent_T_spline(this->u,this->delta_t);
                                T2 tangent_data2[LGT2::DoF];
                                typename Mapper2::Map v2 = Mapper2::map(tangent_data2);
                                v2 = LGT2::vee(Tt);
                                return this->soph->operator()(C0,tangent_data2,residuals);
                            } else if (this->derivative_order==2) {
                                dLGT2 Tt = s.Dt2_parent_T_spline(this->u,this->delta_t);
                                T2 tangent_data2[LGT2::DoF];
                                typename Mapper2::Map v2 = Mapper2::map(tangent_data2);
                                v2 = LGT2::vee(Tt);
                                return this->soph->operator()(C0,tangent_data2,residuals);
                            } else {
                                assert(this->derivative_order < 3); 
                            }
                            return false;
                        }

                };


            template <class ErrorFunctor,int num_residuals_> 
                struct SplineErrorWrapper2 : public SplineErrorWrapper0<ErrorFunctor,num_residuals_> {

                    SplineErrorWrapper2(Sophus::SegmentCase scase, double u, std::shared_ptr<ErrorFunctor> soph, const std::vector<unsigned char> & pmap) : 
                        SplineErrorWrapper0<ErrorFunctor,num_residuals_>(scase, u, soph,pmap) {}

                    SplineErrorWrapper2(unsigned int derivative_order, Sophus::SegmentCase scase, double u, double delta_t, std::shared_ptr<ErrorFunctor> soph, const std::vector<unsigned char> & pmap) : 
                        SplineErrorWrapper0<ErrorFunctor,num_residuals_>(derivative_order, scase, u, delta_t, soph, pmap) {}



                    template <typename T>
                        bool operator()(const T* const C0,const T* const C1,
                                const T* P0, const T* P1, const T* P2, 
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2};
                            const std::vector<unsigned char> & pmap = this->pmap;
                            this->check_map(3);
                            return this->template call<T,T,T, T>(C0, C1, P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],
                                    P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],residuals);
                        }

                    template <typename T>
                        bool operator()(const T* const C0,const T* const C1,
                                const T* P0, const T* P1, const T* P2, const T* P3,
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2,P3};
                            const std::vector<unsigned char> & pmap = this->pmap;
                            this->check_map(4);
                            return this->template call<T,T,T, T>(C0, C1, P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],
                                    P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],residuals);
                        }

                    template <typename T1, typename T2, typename T>
                        bool call(const T1* const C0, const T1* const C1,
                                const T2* const P0, const T2* const P1, const T2* const P2, const T2* const P3, 
                                T* residuals) const {

                            using LGT2 = LieGroup_<T2>;
                            using dLGT2 = typename LieGroup_<T2>::Transformation;
                            // Mapper class is only used to facciliate difference between
                            // SO2 (which uses Scalar as tangent vector type) and other groups
                            // (which use Vector<...> as tangent vector type).
                            using Mapper2 = Mapper<typename LieGroup_<T2>::Tangent>;
                            Sophus::BasisSplineSegment<LGT2> s(this->segment_case,P0,P1,P2,P3);
                            if (this->derivative_order==0) {
                                LGT2 Tt = s.parent_T_spline(this->u);
                                return this->soph->operator()(C0,C1,Tt.data(),residuals);
                            } else if (this->derivative_order==1) {
                                dLGT2 Tt = s.Dt_parent_T_spline(this->u,this->delta_t);
                                T2 tangent_data2[LGT2::DoF];
                                typename Mapper2::Map v2 = Mapper2::map(tangent_data2);
                                v2 = LGT2::vee(Tt);
                                return this->soph->operator()(C0,C1,tangent_data2,residuals);
                            } else if (this->derivative_order==2) {
                                dLGT2 Tt = s.Dt2_parent_T_spline(this->u,this->delta_t);
                                T2 tangent_data2[LGT2::DoF];
                                typename Mapper2::Map v2 = Mapper2::map(tangent_data2);
                                v2 = LGT2::vee(Tt);
                                return this->soph->operator()(C0,C1,tangent_data2,residuals);
                            } else {
                                assert(this->derivative_order < 3); 
                            }
                            return false;
                        }

                };



            //////////////////////////////////////////////////////////////////////////////////////////////
            //
            // Helper functions to insert residual blocks using splines



            // Add an autodiff'ed residual function to a problem defined on a spline. The residual functor is expected to take as 
            // as arguments a class par1, another class par2 and to be estimated at the spline position t on an LieGroup_<Scalar_> class
            // 
            template <class ParamClass1,class ParamClass2,class ErrorFunctor,int num_residuals>
                static bool addResidualFunction2(ceres::Problem &problem, 
                        ParamClass1 & par1, ParamClass2 & par2, 
                        unsigned int derivative_order, double t, double delta_t, 
                        std::shared_ptr<Splined> spline,
                        std::shared_ptr<ErrorFunctor> functor, ceres::LossFunction * loss_function = nullptr) {
                    using Wrapper = SplineErrorWrapper2<ErrorFunctor,num_residuals>;
                    KnotsAndU ku = spline->knots_and_u(t);
                    std::vector<unsigned char> map(4,255);
                    std::map<double *,std::vector<unsigned char>> pmap;
                    pmap[spline->parent_Ts_control_point()[ku.idx_prev].data()].push_back(SPLINE_P0);
                    pmap[spline->parent_Ts_control_point()[ku.idx_0].data()].push_back(SPLINE_P1);
                    pmap[spline->parent_Ts_control_point()[ku.idx_1].data()].push_back(SPLINE_P2);
                    pmap[spline->parent_Ts_control_point()[ku.idx_2].data()].push_back(SPLINE_P3);
                    std::vector<double *> parameter_blocks;
                    parameter_blocks.push_back(par1.data());
                    parameter_blocks.push_back(par2.data()); 
                    for (auto it : pmap) {
                        for (unsigned char x : it.second) {
                            map[x] = parameter_blocks.size()-2;
                        }
                        parameter_blocks.push_back(it.first);
                    }
                    Wrapper * ew = new Wrapper(derivative_order, ku.segment_case,ku.u, delta_t, functor,pmap);
                    ceres::CostFunction *cost_function = NULL;
                    switch (parameter_blocks.size()) {
                        case 5:
                            cost_function =  new ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          ParamClass1::num_parameters,ParamClass2::num_parameters,
                                          LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters> (ew);
                            break;
                        case 6:
                            cost_function =  new ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          ParamClass1::num_parameters,ParamClass2::num_parameters,
                                          LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters> (ew);
                            break;
                        default:
                            assert((parameter_blocks.size()>=5) && (parameter_blocks.size()<=6));
                    }

                    problem.AddResidualBlock(cost_function, loss_function, parameter_blocks);
                    return true;
                }

            template <class ParamClass1,class ErrorFunctor,int num_residuals>
                static bool addResidualFunction1(ceres::Problem &problem, ParamClass1 & par1, 
                        unsigned int derivative_order, double t, double delta_t, 
                        std::shared_ptr<Splined> spline,
                        std::shared_ptr<ErrorFunctor> functor, ceres::LossFunction * loss_function = nullptr) {
                    using Wrapper = SplineErrorWrapper1<ErrorFunctor,num_residuals>;
                    KnotsAndU ku = spline->knots_and_u(t);
                    std::vector<unsigned char> map(4,255);
                    std::map<double *,std::vector<unsigned char>> pmap;
                    pmap[spline->parent_Ts_control_point()[ku.idx_prev].data()].push_back(SPLINE_P0);
                    pmap[spline->parent_Ts_control_point()[ku.idx_0].data()].push_back(SPLINE_P1);
                    pmap[spline->parent_Ts_control_point()[ku.idx_1].data()].push_back(SPLINE_P2);
                    pmap[spline->parent_Ts_control_point()[ku.idx_2].data()].push_back(SPLINE_P3);
                    std::vector<double *> parameter_blocks;
                    parameter_blocks.push_back(par1.data());
                    for (auto it : pmap) {
                        for (unsigned char x : it.second) {
                            map[x] = parameter_blocks.size()-1;
                        }
                        parameter_blocks.push_back(it.first);
                    }
                    Wrapper * ew = new Wrapper(derivative_order, ku.segment_case,ku.u, delta_t, functor, map);
                    ceres::CostFunction *cost_function = NULL;
                    switch (parameter_blocks.size()) {
                        case 4:
                            cost_function =  new ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, ParamClass1::num_parameters,
                                          LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters> (ew);
                            break;
                        case 5:
                            cost_function =  new ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, ParamClass1::num_parameters,
                                          LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters> (ew);
                            break;
                        default:
                            assert((parameter_blocks.size()>=4) && (parameter_blocks.size()<=5));
                    }

                    problem.AddResidualBlock(cost_function, loss_function, parameter_blocks);
                    return true;
                }

            // Add an autodiff'ed residual function to a problem defined on a spline. 
            // derivate order is the derivate order of the spline evaluate (0, 1 or 2).
            // The residual functor is assumed to be defined as 
            //
            // class Functor {
            //  public:
            //      Functor() {}
            //      template <class T>
            //        bool operator()(T const * const P, T* residuals) const {
            //          ...
            //          return true;
            //        }
            // };
            // 
            // Depending on derivative_order P will either be the representation of
            // a LieGroup class (Sophus::SO3d().data()) or its tangent space for 
            // order larger than 0. 
            //
            // The functor will be evaluated at t. delta_t is used as an argument 
            // to the spline derivative for order larger than 0. 
            // 
            // The following functions are provided:
            // - addResidualFunction0(problem, derivative_order, t, delta_t, 
            //      spline, functor, loss_function):
            //      functor defined as above.
            // - addResidualFunction0(problem,t,spline,functor,loss_function): 
            //      equivalent to the previous one, with derivative_order=0.
            //
            // - addResidualFunction1(problem, par1, derivative_order, t, delta_t, 
            //      spline, functor, loss_function):
            //      functor defined with:
            //      template <class T1, class T>
            //        bool operator()(T1 const * const C1, 
            //        T const * const P, T* residuals) const { ... }
            //      where C1 is the representation of par1 (par1.data()), which can
            //      can be a parameter or calibration class.
            // - addResidualFunction1(problem,par1,t,spline,functor,loss_function): 
            //      equivalent to the previous one, with derivative_order=0.
            //
            // - addResidualFunction1(problem, par1, derivative_order, t, delta_t, 
            //      spline, functor, loss_function):
            //      functor defined with:
            //      template <class T1, class T2, class T>
            //        bool operator()(T1 const * const C1, T2 const * const C2, 
            //        T const * const P, T* residuals) const { ... }
            //      where C1 is the representation of par1 (par1.data()), which can
            //      can be a parameter or calibration class, and C2 is similarly
            //      another calibration parameter.
            // - addResidualFunction1(problem,par1,t,spline,functor,loss_function): 
            //      equivalent to the previous one, with derivative_order=0.
            //
            // - addResidualFunction2Points0(problem, derivative_order, t1, t2,
            //      delta_t, spline, functor, loss_function):
            //      functor defined with:
            //      template <class T>
            //        bool operator()(T const * const P, 
            //        T const * const Q, T* residuals) const { ... }
            //      where P and Q are two lie-group representations, sampled on 
            //      the spline at t1 and t2.
            // - addResidualFunction2Points0(problem,t1,t2,spline,functor,loss_function): 
            //      equivalent to the previous one, with derivative_order=0.
            //
            // - addResidual2Functions0(problem, derivative_order, delta_t, 
            //      t1, spline1, t2, spline2, functor, loss_function):
            //      functor defined with:
            //      template <class T>
            //        bool operator()(T const * const P, 
            //        T const * const Q, T* residuals) const { ... }
            //      where P and Q are two lie-group representations, sampled on 
            //      the spline1 at t1 and spline2 at t2.
            // - addResidualFunction2Points0(problem,t1,spline1,t2,spline2,functor,loss_function): 
            //      equivalent to the previous one, with derivative_order=0.
            //
            //
            template <class ErrorFunctor,int num_residuals>
                static bool addResidualFunction0(ceres::Problem &problem, 
                        unsigned int derivative_order, double t, double delta_t, 
                        std::shared_ptr<Splined> spline,
                        std::shared_ptr<ErrorFunctor> functor, ceres::LossFunction * loss_function = nullptr) {
                    using Wrapper = SplineErrorWrapper0<ErrorFunctor,num_residuals>;
                    KnotsAndU ku = spline->knots_and_u(t);
                    std::vector<unsigned char> map(4,255);
                    std::map<double *,std::vector<unsigned char>> pmap;
                    pmap[spline->parent_Ts_control_point()[ku.idx_prev].data()].push_back(SPLINE_P0);
                    pmap[spline->parent_Ts_control_point()[ku.idx_0].data()].push_back(SPLINE_P1);
                    pmap[spline->parent_Ts_control_point()[ku.idx_1].data()].push_back(SPLINE_P2);
                    pmap[spline->parent_Ts_control_point()[ku.idx_2].data()].push_back(SPLINE_P3);
                    std::vector<double *> parameter_blocks;
                    for (auto it : pmap) {
                        for (unsigned char x : it.second) {
                            map[x] = parameter_blocks.size();
                        }
                        parameter_blocks.push_back(it.first);
                    }
                    Wrapper * ew = new Wrapper(derivative_order, ku.segment_case,ku.u, delta_t, functor, map);
                    ceres::CostFunction *cost_function = NULL;
                    switch (parameter_blocks.size()) {
                        case 3:
                            cost_function =  new ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters> (ew);
                            break;
                        case 4:
                            cost_function =  new ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters> (ew);
                            break;
                        default:
                            assert((parameter_blocks.size()>=3) && (parameter_blocks.size()<=4));
                    }

                    problem.AddResidualBlock(cost_function, loss_function, parameter_blocks);
                    return true;
                }

            template <class ErrorFunctor,int num_residuals>
                static bool addResidualFunction2Points0(ceres::Problem &problem, 
                        unsigned int derivative_order, double t1, double t2, double delta_t, 
                        std::shared_ptr<Splined> spline,
                        std::shared_ptr<ErrorFunctor> functor, ceres::LossFunction * loss_function = nullptr) {
                    using Wrapper = SplineError2PointsWrapper0<ErrorFunctor,num_residuals>;
                    KnotsAndU ku1 = spline->knots_and_u(t1);
                    KnotsAndU ku2 = spline->knots_and_u(t2);
                    std::vector<unsigned char> map(8,255);
                    std::map<double *,std::vector<unsigned char>> pmap;
                    pmap[spline->parent_Ts_control_point()[ku1.idx_prev].data()].push_back(SPLINE_P0);
                    pmap[spline->parent_Ts_control_point()[ku1.idx_0].data()].push_back(SPLINE_P1);
                    pmap[spline->parent_Ts_control_point()[ku1.idx_1].data()].push_back(SPLINE_P2);
                    pmap[spline->parent_Ts_control_point()[ku1.idx_2].data()].push_back(SPLINE_P3);
                    pmap[spline->parent_Ts_control_point()[ku2.idx_prev].data()].push_back(SPLINE_Q0);
                    pmap[spline->parent_Ts_control_point()[ku2.idx_0].data()].push_back(SPLINE_Q1);
                    pmap[spline->parent_Ts_control_point()[ku2.idx_1].data()].push_back(SPLINE_Q2);
                    pmap[spline->parent_Ts_control_point()[ku2.idx_2].data()].push_back(SPLINE_Q3);
                    std::vector<double *> parameter_blocks;
                    for (auto it : pmap) {
                        for (unsigned char x : it.second) {
                            map[x] = parameter_blocks.size();
                        }
                        parameter_blocks.push_back(it.first);
                    }
                    Wrapper * ew = new Wrapper(derivative_order, 
                            ku1.segment_case,ku1.u, ku2.segment_case,ku2.u, delta_t, functor, pmap);
                    ceres::CostFunction *cost_function = NULL;
                    switch (parameter_blocks.size()) {
                        case 3:
                            cost_function =  new ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters> (ew);
                            break;
                        case 4:
                            cost_function =  new ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters> (ew);
                            break;
                        case 5:
                            cost_function =  new ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters,
                                          LieGroupd::num_parameters > (ew);
                            break;
                        case 6:
                            cost_function =  new ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters,
                                          LieGroupd::num_parameters, LieGroupd::num_parameters > (ew);
                            break;
                        case 7:
                            cost_function =  new ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters,
                                          LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters > (ew);
                            break;
                        case 8:
                            cost_function =  new ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters,
                                          LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters > (ew);
                            break;
                        default:
                            assert((parameter_blocks.size()>=3) && (parameter_blocks.size()<=8));
                    }

                    problem.AddResidualBlock(cost_function, loss_function, parameter_blocks);

                    return true;
                }

            template <class ErrorFunctor,int num_residuals>
                static bool addResidual2Functions0(ceres::Problem &problem, 
                        unsigned int derivative_order, 
                        double delta_t1, double t1, std::shared_ptr<Splined> spline1,
                        double delta_t2, double t2, std::shared_ptr<Splined> spline2,
                        std::shared_ptr<ErrorFunctor> functor, ceres::LossFunction * loss_function = nullptr) {
                    using Wrapper = SplineError2FunctionsWrapper0<ErrorFunctor,num_residuals>;
                    KnotsAndU ku1 = spline1->knots_and_u(t1);
                    KnotsAndU ku2 = spline2->knots_and_u(t2);
                    std::vector<unsigned char> map(8,255);
                    std::map<double *,std::vector<unsigned char>> pmap;
                    pmap[spline1->parent_Ts_control_point()[ku1.idx_prev].data()].push_back(SPLINE_P0);
                    pmap[spline1->parent_Ts_control_point()[ku1.idx_0].data()].push_back(SPLINE_P1);
                    pmap[spline1->parent_Ts_control_point()[ku1.idx_1].data()].push_back(SPLINE_P2);
                    pmap[spline1->parent_Ts_control_point()[ku1.idx_2].data()].push_back(SPLINE_P3);
                    pmap[spline2->parent_Ts_control_point()[ku2.idx_prev].data()].push_back(SPLINE_Q0);
                    pmap[spline2->parent_Ts_control_point()[ku2.idx_0].data()].push_back(SPLINE_Q1);
                    pmap[spline2->parent_Ts_control_point()[ku2.idx_1].data()].push_back(SPLINE_Q2);
                    pmap[spline2->parent_Ts_control_point()[ku2.idx_2].data()].push_back(SPLINE_Q3);
                    std::vector<double *> parameter_blocks;
                    for (auto it : pmap) {
                        for (unsigned char x : it.second) {
                            map[x] = parameter_blocks.size();
                        }
                        parameter_blocks.push_back(it.first);
                    }
                    Wrapper * ew = new Wrapper(derivative_order, 
                            ku1.segment_case,ku1.u, ku2.segment_case,ku2.u, 
                            delta_t1, delta_t2, functor, pmap);
                    ceres::CostFunction *cost_function = NULL;
                    switch (parameter_blocks.size()) {
                        case 6:
                            cost_function =  new ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters,
                                          LieGroupd::num_parameters, LieGroupd::num_parameters > (ew);
                            break;
                        case 7:
                            cost_function =  new ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters,
                                          LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters > (ew);
                            break;
                        case 8:
                            cost_function =  new ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters,
                                          LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters, LieGroupd::num_parameters > (ew);
                            break;
                        default:
                            assert((parameter_blocks.size()>=6) && (parameter_blocks.size()<=8));
                    }

                    problem.AddResidualBlock(cost_function, loss_function, parameter_blocks);

                    return true;
                }

            template <class ParamClass1,class ParamClass2,class ErrorFunctor,int num_residuals>
                static bool addResidualFunction2(ceres::Problem &problem, 
                        ParamClass1 & par1, ParamClass2 & par2, 
                        double t,
                        std::shared_ptr<Splined> spline,
                        std::shared_ptr<ErrorFunctor> functor, ceres::LossFunction * loss_function = nullptr) {
                    return addResidualFunction2<ParamClass1,ParamClass2,ErrorFunctor,num_residuals>(problem, par1, par2, 0, t, 0.0, spline, functor, loss_function); 
                }

            template <class ParamClass1,class ErrorFunctor,int num_residuals>
                static bool addResidualFunction1(ceres::Problem &problem, ParamClass1 & par1, 
                        double t,
                        std::shared_ptr<Splined> spline,
                        std::shared_ptr<ErrorFunctor> functor, ceres::LossFunction * loss_function = nullptr) {
                    return addResidualFunction1<ParamClass1,ErrorFunctor,num_residuals>(problem, par1, 0, t, 0.0, spline, functor, loss_function); 
                }

            template <class ErrorFunctor,int num_residuals>
                static bool addResidualFunction0(ceres::Problem &problem, 
                        double t,
                        std::shared_ptr<Splined> spline,
                        std::shared_ptr<ErrorFunctor> functor, ceres::LossFunction * loss_function = nullptr) {
                    return addResidualFunction0<ErrorFunctor,num_residuals>(problem, 0, t, 0.0, spline, functor, loss_function); 
                }

            template <class ErrorFunctor,int num_residuals>
                static bool addResidualFunction2Points0(ceres::Problem &problem, 
                        double t1, double t2, 
                        std::shared_ptr<Splined> spline,
                        std::shared_ptr<ErrorFunctor> functor, ceres::LossFunction * loss_function = nullptr) {
                    return addResidualFunction2Points0<ErrorFunctor,num_residuals>(problem, 0, t1, t2, 0.0, spline, functor, loss_function); 
                }

            template <class ErrorFunctor, int num_residuals>
                static bool addResidual2Functions0(ceres::Problem &problem, 
                        double t1, std::shared_ptr<Splined> spline1,
                        double t2, std::shared_ptr<Splined> spline2,
                        std::shared_ptr<ErrorFunctor> functor, ceres::LossFunction * loss_function = nullptr) {
                    return addResidual2Functions0<ErrorFunctor,num_residuals>(problem, 0, 
                            0.0, t1, spline1, 0.0, t2, spline2, functor, loss_function); 
                }
        };

}

