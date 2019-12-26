#pragma once
#include "hnswlib.h"

namespace hnswlib {
	static float
		Cosine(const void* pVect1, const void* pVect2, const void* qty_ptr) {
		size_t qty = *((size_t*)qty_ptr);
		float res1 = 0;
		float res2 = 0;
		float res12 = 0;
		for (unsigned i = 0; i < qty; i++) {
			res1 += ((float*)pVect1)[i] * ((float*)pVect1)[i];
			res2 += ((float*)pVect2)[i] * ((float*)pVect2)[i];
			res12 += ((float*)pVect1)[i] * ((float*)pVect2)[i];			
		}

		float res = 1.0 - res12 / (sqrt(res1 * res2));
		return (res);

	}

#if defined(USE_AVX)

	static float
		CosineSIMD16Ext(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
		float* pVect1 = (float*)pVect1v;
		float* pVect2 = (float*)pVect2v;
		size_t qty = *((size_t*)qty_ptr);
		float PORTABLE_ALIGN32 TmpRes12[8];
		float PORTABLE_ALIGN32 TmpRes1[8];
		float PORTABLE_ALIGN32 TmpRes2[8];
		size_t qty16 = qty >> 4;

		const float* pEnd1 = pVect1 + (qty16 << 4);

		__m256 v1, v2;
		__m256 sum12 = _mm256_set1_ps(0);
		__m256 sum1 = _mm256_set1_ps(0);
		__m256 sum2 = _mm256_set1_ps(0);

		while (pVect1 < pEnd1) {
			v1 = _mm256_loadu_ps(pVect1);
			pVect1 += 8;
			v2 = _mm256_loadu_ps(pVect2);
			pVect2 += 8;
			sum12 = _mm256_add_ps(sum12, _mm256_mul_ps(v1, v2));
			sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(v1, v1));
			sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(v2, v2));


			v1 = _mm256_loadu_ps(pVect1);
			pVect1 += 8;
			v2 = _mm256_loadu_ps(pVect2);
			pVect2 += 8;
			sum12 = _mm256_add_ps(sum12, _mm256_mul_ps(v1, v2));
			sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(v1, v1));
			sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(v2, v2));
	}

		_mm256_store_ps(TmpRes12, sum12);
		_mm256_store_ps(TmpRes1, sum1);
		_mm256_store_ps(TmpRes2, sum2);
		float res12 = TmpRes12[0] + TmpRes12[1] + TmpRes12[2] + TmpRes12[3] + TmpRes12[4] + TmpRes12[5] + TmpRes12[6] + TmpRes12[7];
		float res1 = TmpRes1[0] + TmpRes1[1] + TmpRes1[2] + TmpRes1[3] + TmpRes1[4] + TmpRes1[5] + TmpRes1[6] + TmpRes1[7];
		float res2 = TmpRes2[0] + TmpRes2[1] + TmpRes2[2] + TmpRes2[3] + TmpRes2[4] + TmpRes2[5] + TmpRes2[6] + TmpRes2[7];

		float res = 1.0 - res12 / (sqrt(res1 * res2));

		return (res);
	}
#elif defined(USE_SSE)
	static float
		CosineSIMD16Ext(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
		float* pVect1 = (float*)pVect1v;
		float* pVect2 = (float*)pVect2v;
		size_t qty = *((size_t*)qty_ptr);
		float PORTABLE_ALIGN32 TmpRes12[8];
		float PORTABLE_ALIGN32 TmpRes1[8];
		float PORTABLE_ALIGN32 TmpRes2[8];
		size_t qty16 = qty >> 4;

		const float* pEnd1 = pVect1 + (qty16 << 4);

		__m128 v1, v2;
		__m128 sum12 = _mm_set1_ps(0);
		__m128 sum1 = _mm_set1_ps(0);
		__m128 sum2 = _mm_set1_ps(0);

		while (pVect1 < pEnd1) {
			v1 = _mm_loadu_ps(pVect1);
			pVect1 += 4;
			v2 = _mm_loadu_ps(pVect2);
			pVect2 += 4;
			sum12 = _mm_add_ps(sum12, _mm_mul_ps(v1, v2));
			sum1 = _mm_add_ps(sum1, _mm_mul_ps(v1, v1));
			sum2 = _mm_add_ps(sum2, _mm_mul_ps(v2, v2));

			v1 = _mm_loadu_ps(pVect1);
			pVect1 += 4;
			v2 = _mm_loadu_ps(pVect2);
			pVect2 += 4;
			sum12 = _mm_add_ps(sum12, _mm_mul_ps(v1, v2));
			sum1 = _mm_add_ps(sum1, _mm_mul_ps(v1, v1));
			sum2 = _mm_add_ps(sum2, _mm_mul_ps(v2, v2));

			v1 = _mm_loadu_ps(pVect1);
			pVect1 += 4;
			v2 = _mm_loadu_ps(pVect2);
			pVect2 += 4;
			sum12 = _mm_add_ps(sum12, _mm_mul_ps(v1, v2));
			sum1 = _mm_add_ps(sum1, _mm_mul_ps(v1, v1));
			sum2 = _mm_add_ps(sum2, _mm_mul_ps(v2, v2));

			v1 = _mm_loadu_ps(pVect1);
			pVect1 += 4;
			v2 = _mm_loadu_ps(pVect2);
			pVect2 += 4;
			sum12 = _mm_add_ps(sum12, _mm_mul_ps(v1, v2));
			sum1 = _mm_add_ps(sum1, _mm_mul_ps(v1, v1));
			sum2 = _mm_add_ps(sum2, _mm_mul_ps(v2, v2));
		}

		_mm_store_ps(TmpRes12, sum12);
		_mm_store_ps(TmpRes1, sum1);
		_mm_store_ps(TmpRes2, sum2);
		float res12 = TmpRes12[0] + TmpRes12[1] + TmpRes12[2] + TmpRes12[3];
		float res1 = TmpRes1[0] + TmpRes1[1] + TmpRes1[2] + TmpRes1[3];
		float res2 = TmpRes2[0] + TmpRes2[1] + TmpRes2[2] + TmpRes2[3];

		float res = 1.0 - res12 / (sqrt(res1 * res2));

		return (res);
	}
#endif

#ifdef USE_SSE
	static float
		CosineSIMD4Ext(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
		float PORTABLE_ALIGN32 TmpRes12[8];
		float PORTABLE_ALIGN32 TmpRes1[8];
		float PORTABLE_ALIGN32 TmpRes2[8];
		float* pVect1 = (float*)pVect1v;
		float* pVect2 = (float*)pVect2v;
		size_t qty = *((size_t*)qty_ptr);

		size_t qty16 = qty >> 2;

		const float* pEnd1 = pVect1 + (qty16 << 2);

		__m128 v1, v2;
		__m128 sum12 = _mm_set1_ps(0);
		__m128 sum1 = _mm_set1_ps(0);
		__m128 sum2 = _mm_set1_ps(0);

		while (pVect1 < pEnd1) {
			v1 = _mm_loadu_ps(pVect1);
			pVect1 += 4;
			v2 = _mm_loadu_ps(pVect2);
			pVect2 += 4;
			sum12 = _mm_add_ps(sum12, _mm_mul_ps(v1, v2));
			sum1 = _mm_add_ps(sum1, _mm_mul_ps(v1, v1));
			sum2 = _mm_add_ps(sum2, _mm_mul_ps(v2, v2));

			v1 = _mm_loadu_ps(pVect1);
			pVect1 += 4;
			v2 = _mm_loadu_ps(pVect2);
			pVect2 += 4;
			sum12 = _mm_add_ps(sum12, _mm_mul_ps(v1, v2));
			sum1 = _mm_add_ps(sum1, _mm_mul_ps(v1, v1));
			sum2 = _mm_add_ps(sum2, _mm_mul_ps(v2, v2));
		}

		_mm_store_ps(TmpRes12, sum12);
		_mm_store_ps(TmpRes1, sum1);
		_mm_store_ps(TmpRes2, sum2);
		float res12 = TmpRes12[0] + TmpRes12[1] + TmpRes12[2] + TmpRes12[3];
		float res1 = TmpRes1[0] + TmpRes1[1] + TmpRes1[2] + TmpRes1[3];
		float res2 = TmpRes2[0] + TmpRes2[1] + TmpRes2[2] + TmpRes2[3];

		float res = 1.0 - res12 / (sqrt(res1 * res2));

		return (res);

	}
#endif

	class CosSpace : public SpaceInterface<float> {

		DISTFUNC<float> fstdistfunc_;
		size_t data_size_;
		size_t dim_;
	public:
		CosSpace()
		{
			dim_ = 0;
			data_size_ = 0;
			fstdistfunc_ = Cosine;
		}

		void SetDim(size_t dim)
		{
			fstdistfunc_ = Cosine;
#if defined(USE_SSE) || defined(USE_AVX)
			if (dim % 4 == 0)
				fstdistfunc_ = CosineSIMD4Ext;
			if (dim % 16 == 0)
				fstdistfunc_ = CosineSIMD16Ext;
			/*else{
				throw runtime_error("Data type not supported!");
			}*/
#endif
			dim_ = dim;
			data_size_ = dim * sizeof(float);
		}

		CosSpace(size_t dim) {
			fstdistfunc_ = Cosine;
#if defined(USE_SSE) || defined(USE_AVX)
			if (dim % 4 == 0)
				fstdistfunc_ = CosineSIMD4Ext;
			if (dim % 16 == 0)
				fstdistfunc_ = CosineSIMD16Ext;
			/*else{
				throw runtime_error("Data type not supported!");
			}*/
#endif
			dim_ = dim;
			data_size_ = dim * sizeof(float);
		}

		size_t get_data_size() {
			return data_size_;
		}

		DISTFUNC<float> get_dist_func() {
			return fstdistfunc_;
		}

		void* get_dist_func_param() {
			return &dim_;
		}

		~CosSpace() {}
	};


}
