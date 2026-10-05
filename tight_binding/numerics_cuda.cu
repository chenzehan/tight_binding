#include "numerics_cuda.cuh"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include <iostream>

using Complex = std::complex<double>;

struct DenseRealOnDevice {
	cusparseDnMatDescr_t descr;
	int rows, cols;
	double* data;
	DenseRealOnDevice(const Eigen::MatrixXd& M) {
		rows = M.rows();
		cols = M.cols();
		cudaMalloc(&data, rows * cols * sizeof(double));
		cudaMemcpy(data, M.data(), rows * cols * sizeof(double), cudaMemcpyHostToDevice);
		cusparseCreateDnMat(&descr, rows, cols, rows, data, CUDA_R_64F, CUSPARSE_ORDER_COL);
	}
	DenseRealOnDevice(int rows, int cols) : rows(rows), cols(cols) {
		cudaMalloc(&data, rows * cols * sizeof(double));
		cusparseCreateDnMat(&descr, rows, cols, rows, data, CUDA_R_64F, CUSPARSE_ORDER_COL);
	}
	DenseRealOnDevice(DenseRealOnDevice&& M) noexcept {
		rows = M.rows;
		cols = M.cols;
		data = M.data;
		descr = M.descr;
		M.data = nullptr;
	}
	~DenseRealOnDevice() {
		cudaFree(data);
		cusparseDestroyDnMat(descr);
	}
	void swap(DenseRealOnDevice& M) {
		std::swap(rows, M.rows);
		std::swap(cols, M.cols);
		std::swap(data, M.data);
		std::swap(descr, M.descr);
	}
};

struct SparseRealCscOnDevice {
	cusparseSpMatDescr_t descr;
	int rows, cols, nnz;
	int* col_offset;
	int* row_indices;
	double* values;
	// assume M is compressed
	SparseRealCscOnDevice(const Eigen::SparseMatrix<double>& M) {
		rows = M.rows();
		cols = M.cols();
		nnz = M.nonZeros();
		cudaMalloc(&col_offset, (cols + 1) * sizeof(int));
		cudaMalloc(&row_indices, nnz * sizeof(int));
		cudaMalloc(&values, nnz * sizeof(double));
		cudaMemcpy(col_offset, M.outerIndexPtr(), (cols + 1) * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(row_indices, M.innerIndexPtr(), nnz * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(values, M.valuePtr(), nnz * sizeof(double), cudaMemcpyHostToDevice);
		cusparseCreateCsc(&descr, rows, cols, nnz,
			col_offset, row_indices, values,
			CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
	}
	~SparseRealCscOnDevice() {
		cudaFree(col_offset);
		cudaFree(row_indices);
		cudaFree(values);
		cusparseDestroySpMat(descr);
	}
	// C = self * B
	cusparseStatus_t mul(const DenseRealOnDevice& B, DenseRealOnDevice& C,
		cusparseHandle_t handle, void* dBuffer) {

		double alpha = 1.0;
		double beta = 0.0;
		cusparseStatus_t status;
		status = cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha, this->descr, B.descr, &beta, C.descr,
			CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
		return status;
	}
	// C = alpha * self * B + beta * C
	cusparseStatus_t mul_add(const DenseRealOnDevice& B, DenseRealOnDevice& C,
		double alpha, double beta, cusparseHandle_t handle, void* dBuffer) {

		cusparseStatus_t status;
		status = cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha, this->descr, B.descr, &beta, C.descr,
			CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
		return status;
	}
};

struct DenseComplexOnDevice {
	cusparseDnMatDescr_t descr;
	int rows, cols;
	cuDoubleComplex* data;
	DenseComplexOnDevice(const Eigen::MatrixXcd& M) {
		rows = M.rows();
		cols = M.cols();
		cudaMalloc(&data, rows * cols * sizeof(cuDoubleComplex));
		cudaMemcpy(data, M.data(), rows * cols * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
		cusparseCreateDnMat(&descr, rows, cols, rows, data, CUDA_C_64F, CUSPARSE_ORDER_COL);
	}
	DenseComplexOnDevice(int rows, int cols) : rows(rows), cols(cols) {
		cudaMalloc(&data, rows * cols * sizeof(cuDoubleComplex));
		cusparseCreateDnMat(&descr, rows, cols, rows, data, CUDA_C_64F, CUSPARSE_ORDER_COL);
	}
	DenseComplexOnDevice(DenseComplexOnDevice&& M) noexcept {
		rows = M.rows;
		cols = M.cols;
		data = M.data;
		descr = M.descr;
		M.data = nullptr;
	}
	~DenseComplexOnDevice() {
		cudaFree(data);
		cusparseDestroyDnMat(descr);
	}
	void swap(DenseComplexOnDevice& M) {
		std::swap(rows, M.rows);
		std::swap(cols, M.cols);
		std::swap(data, M.data);
		std::swap(descr, M.descr);
	}
};

struct SparseComplexCscOnDevice {
	cusparseSpMatDescr_t descr;
	int rows, cols, nnz;
	int* col_offset;
	int* row_indices;
	cuDoubleComplex* values;
	// assume M is compressed
	SparseComplexCscOnDevice(const Eigen::SparseMatrix<Complex>& M) {
		rows = M.rows();
		cols = M.cols();
		nnz = M.nonZeros();
		cudaMalloc(&col_offset, (cols + 1) * sizeof(int));
		cudaMalloc(&row_indices, nnz * sizeof(int));
		cudaMalloc(&values, nnz * sizeof(cuDoubleComplex));
		cudaMemcpy(col_offset, M.outerIndexPtr(), (cols + 1) * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(row_indices, M.innerIndexPtr(), nnz * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(values, M.valuePtr(), nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
		cusparseCreateCsc(&descr, rows, cols, nnz,
			col_offset, row_indices, values,
			CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);
	}
	~SparseComplexCscOnDevice() {
		cudaFree(col_offset);
		cudaFree(row_indices);
		cudaFree(values);
		cusparseDestroySpMat(descr);
	}
	// C = self * B
	cusparseStatus_t mul(const DenseComplexOnDevice& B, DenseComplexOnDevice& C,
		cusparseHandle_t handle, void* dBuffer) {

		cuDoubleComplex alpha{ 1.0, 0.0 };
		cuDoubleComplex beta{ 0.0, 0.0 };
		cusparseStatus_t status;
		status = cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha, this->descr, B.descr, &beta, C.descr,
			CUDA_C_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
		return status;
	}
	// C = alpha * self * B + beta * C
	cusparseStatus_t mul_add(const DenseComplexOnDevice& B, DenseComplexOnDevice& C,
		cuDoubleComplex alpha, cuDoubleComplex beta, cusparseHandle_t handle, void* dBuffer) {

		cusparseStatus_t status;
		status = cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha, this->descr, B.descr, &beta, C.descr,
			CUDA_C_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
		return status;
	}
};

namespace cuda {
	double trace_braket_real(const DenseRealOnDevice& bras_T, const DenseRealOnDevice& kets,
		cublasHandle_t handle) {
		double trace = 0.0;
		cublasDdot_v2(handle, bras_T.rows * bras_T.cols, 
			bras_T.data, 1, kets.data, 1, &trace);
		return trace;
	}
	// M must be made compressed
	std::vector<double> Cheb_first_kind_seq_braket(const Eigen::SparseMatrix<double>& M, int n,
		const Eigen::MatrixXcd& bras_T, const Eigen::MatrixXcd& kets)
	{
		std::cout << "Real Cheb on CUDA" << std::endl;
		if (n < 2) {
			std::cerr << "n must be greater than or equal to 2" << std::endl;
			return {};
		}
		std::vector<double> mu_n(n);
		SparseRealCscOnDevice M_dev(M);
		DenseRealOnDevice bras_T_re_dev(bras_T.real());
		DenseRealOnDevice bras_T_im_dev(bras_T.imag());
		DenseRealOnDevice Tn_2_re(kets.real());
		DenseRealOnDevice Tn_2_im(kets.imag());
		DenseRealOnDevice Tn_1_re(kets.rows(), kets.cols());
		DenseRealOnDevice Tn_1_im(kets.rows(), kets.cols());
		cusparseHandle_t handle;
		cusparseCreate(&handle);
		cublasHandle_t cublasHandle;
		cublasCreate_v2(&cublasHandle);
		double alpha_0 = 1.0;
		double beta_0 = 0.0;
		size_t bufferSize, bufferSize_1, bufferSize_2;
		void* dBuffer = nullptr;
		cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha_0, M_dev.descr, Tn_2_re.descr, &beta_0, bras_T_re_dev.descr,
			CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize_1);

		double alpha = 2.0;
		double beta = -1.0;
		cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha, M_dev.descr, Tn_2_re.descr, &beta, bras_T_re_dev.descr,
			CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize_2);
		bufferSize = std::max(bufferSize_1, bufferSize_2);
		cudaMalloc(&dBuffer, bufferSize);

		// Tn_1 = M * Tn_2
		M_dev.mul(Tn_2_re, Tn_1_re, handle, dBuffer);
		M_dev.mul(Tn_2_im, Tn_1_im, handle, dBuffer);
		mu_n[0] = trace_braket_real(bras_T_re_dev, Tn_2_re, cublasHandle)
				+ trace_braket_real(bras_T_im_dev, Tn_2_im, cublasHandle);
		mu_n[1] = trace_braket_real(bras_T_re_dev, Tn_1_re, cublasHandle)
				+ trace_braket_real(bras_T_im_dev, Tn_1_im, cublasHandle);
		for (int i = 2; i < n; i++) {
			// Tn_2 = 2 * M * Tn_1 - Tn_2
			M_dev.mul_add(Tn_1_re, Tn_2_re, alpha, beta, handle, dBuffer);
			M_dev.mul_add(Tn_1_im, Tn_2_im, alpha, beta, handle, dBuffer);
			/*cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
				&alpha, M_dev.descr, Tn_1_re.descr, &beta, Tn_2_re.descr,
				CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
			cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
				&alpha, M_dev.descr, Tn_1_im.descr, &beta, Tn_2_im.descr,
				CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);*/
			mu_n[i] = trace_braket_real(bras_T_re_dev, Tn_2_re, cublasHandle)
					+ trace_braket_real(bras_T_im_dev, Tn_2_im, cublasHandle);
			std::cout << i << " " << mu_n[i] << std::endl;
			Tn_1_re.swap(Tn_2_re);
			Tn_1_im.swap(Tn_2_im);
		}

		cudaFree(dBuffer);
		cusparseDestroy(handle);
		cublasDestroy_v2(cublasHandle);
		
		return mu_n;
	}

	double trace_braket_complex(const DenseComplexOnDevice& bras_T, const DenseComplexOnDevice& kets,
		cublasHandle_t handle) {
		cuDoubleComplex trace{ 0.0, 0.0 };
		// TO check
		cublasZdotc_v2(handle, bras_T.rows * bras_T.cols,
			bras_T.data, 1, kets.data, 1, &trace);
		return trace.x;
	}

	// M must be made compressed
	std::vector<double> Cheb_first_kind_seq_braket(const Eigen::SparseMatrix<Complex>& M, int n,
		const Eigen::MatrixXcd& bras_T, const Eigen::MatrixXcd& kets)
	{
		std::cout << "Complex Cheb on CUDA" << std::endl;
		if (n < 2) {
			std::cerr << "n must be greater than or equal to 2" << std::endl;
			return {};
		}
		std::vector<double> mu_n(n);
		SparseComplexCscOnDevice M_dev(M); // M on device
		DenseComplexOnDevice bras_T_dev(bras_T);
		DenseComplexOnDevice Tn_2(kets);
		DenseComplexOnDevice Tn_1(kets.rows(), kets.cols());
		cusparseHandle_t handle;
		cusparseCreate(&handle);
		cublasHandle_t cublasHandle;
		cublasCreate_v2(&cublasHandle);
		cuDoubleComplex alpha_0{ 2.0, 0.0 };
		cuDoubleComplex beta_0{ -1.0, 0.0 };
		size_t bufferSize, bufferSize_1, bufferSize_2;
		void* dBuffer = nullptr;
		cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha_0, M_dev.descr, Tn_2.descr, &beta_0, bras_T_dev.descr,
			CUDA_C_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize_1);

		cuDoubleComplex alpha{ 2.0, 0.0 };
		cuDoubleComplex beta{ -1.0, 0.0 };
		cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha, M_dev.descr, Tn_2.descr, &beta, bras_T_dev.descr,
			CUDA_C_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize_2);
		bufferSize = std::max(bufferSize_1, bufferSize_2);
		cudaMalloc(&dBuffer, bufferSize);

		// Tn_1 = M * Tn_2
		auto st = M_dev.mul(Tn_2, Tn_1, handle, dBuffer);
		mu_n[0] = trace_braket_complex(bras_T_dev, Tn_2, cublasHandle);
		mu_n[1] = trace_braket_complex(bras_T_dev, Tn_1, cublasHandle);
		std::cout << 0 << " " << mu_n[0] << std::endl;
		std::cout << 1 << " " << mu_n[1] << std::endl;
		for (int i = 2; i < n; i++) {
			// Tn_2 = 2 * M * Tn_1 - Tn_2
			cusparseStatus_t status;
			status = M_dev.mul_add(Tn_1, Tn_2, alpha, beta, handle, dBuffer);
			/*status = cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
				&alpha, M_dev.descr, Tn_1.descr, &beta, Tn_2.descr,
				CUDA_C_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);*/
			mu_n[i] = trace_braket_complex(bras_T_dev, Tn_2, cublasHandle);
			std::cout << i << " " << mu_n[i] << std::endl;
			Tn_1.swap(Tn_2);
		}

		cudaFree(dBuffer);
		cusparseDestroy(handle);
		cublasDestroy_v2(cublasHandle);

		return mu_n;
	}
}
