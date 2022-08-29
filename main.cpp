/* 
 * @theme Inplement of Distortion-Free Wide-Angle Portraits on Camera Phones
 * @ahthor hzzg
 * @date 2022.05
 */

# include <Eigen/Core>
# include <opencv2/highgui/highgui.hpp>
# include <opencv2/core/core.hpp>
# include <opencv2/imgproc/imgproc.hpp>
# include <string>
# include <vector>
# include <iostream>
# include <cmath>
# include <algorithm>
# include <ceres/ceres.h>
# include <glog/logging.h>
# include <fstream>


// function declarations
void onMouseClick(int event, int x, int y, int flags, void* param);
void onMouseGetRegion(int event, int x, int y, int flags, void* param);
cv::Mat getMesh(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> meshx,
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> meshy,
	bool withLine = true);
cv::Mat stretch(cv::Mat imOrigin,
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> meshx,
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> meshy);
cv::Mat stretchMask(cv::Mat imOrigin,
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> meshx,
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> meshy,
	cv::Mat mask);
double Kh(double x);
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> cutPadding(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat,
	int padding);
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> extendPadding(
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat,
	int padding);
double angle2radian(double angle);
double m(double r, double ra, double rb);
template <typename T>
void writeMatrix(std::string filePath, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat);
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> readMatrix(std::string filePath);


// global variables
// interaction
cv::Point2i prePoint(-1, -1);
std::string winGetRegion = "Choose Mask Region";
// mask
cv::Mat mask;
// algorithm params
int k = 0;  // the number of selected regions
int scale;  // from image pixels to mesh
// weights for cost functions (sqrt (end without 2) or not sqrt (end with 2) is for convenience of calculation)
const double lambdaF = sqrt(4);  // 4
const double lambdaB2 = 2.;  // 2
const double lambdaR2 = 0.5;  // 0.5
const double lambdaA = sqrt(4);  // 4


// cost functions
// Ef part 1
struct costEf1 
{
	costEf1(double weight_, double x_, double y_, double lambda_ = lambdaF) : weight(weight_), ux(x_), uy(y_), lambda(lambda_) { }

	template <typename T>
	bool operator()(const T* const x, const T* const y, 
		const T* const a, const T* const b, const T* const tx, const T* const ty, 
		T* residual) const 
	{
		double weightTemp = ceres::sqrt(weight) * lambda;
		residual[0] = weightTemp * (x[0] - (a[0] * ux + b[0] * uy + tx[0]));
		residual[1] = weightTemp * (y[0] - (-b[0] * ux + a[0] * uy + ty[0]));
		return true;
	}

private:
	const double weight;
	const double ux;
	const double uy;
	const double lambda;
};

// Ef part2
struct costEf2 
{
	costEf2(double ws_, double st_, double lambda_ = lambdaF) : ws(ws_), st(st_), lambda(lambda_) { }

	template <typename T>
	bool operator()(const T* const a, T* residual) const
	{
		residual[0] = lambda * ceres::sqrt(ws) * (a[0] - st);
		return true;
	}

private:
	const double ws;
	const double st;
	const double lambda;
};

// Eb and Er are combined
struct costEb
{
	costEb(double lambdaB_ = ::lambdaB2, double lambdaR_ = ::lambdaR2) : lambdaB2(lambdaB_), lambdaR2(lambdaR_) { }
	template <typename T>
	bool operator()(const T* const vi, const T* const vj, T* residual) const
	{
		residual[0] = ceres::sqrt(lambdaB2 + lambdaR2) * (vi[0] - vj[0]);
		return true;
	}

private:
	const double lambdaB2;
	const double lambdaR2;
};

// Ea Greater than version
struct costEaG
{
	costEaG(double ref_, double lambda_ = lambdaA) : ref(ref_), lambda(lambda_) { }
	template <typename T>
	bool operator()(const T* const x, T* residual) const
	{
		if (x[0] > ref) {
			residual[0] = lambda * (x[0] - ref);
		}
		else {
			residual[0] = T(0.0);
		}
		return true;
	}

private:
	const double ref;
	const double lambda;
};

// Ea Less than version
struct costEaL
{
	costEaL(double ref_, double lambda_ = lambdaA) : ref(ref_), lambda(lambda_) { }
	template <typename T>
	bool operator()(const T* const x, T* residual) const
	{
		if (x[0] < ref) {
			residual[0] = lambda * (x[0] - ref);
		}
		else {
			residual[0] = T(0.0);
		}
		return true;
	}

private:
	const double ref;
	const double lambda;
};


int main()
{
	// parameters
	// paths
	std::string path = "D:/myC/CppProject/WideAnglePortrait/WideAnglePortrait";
	std::string imagePath = path + "/images";
	std::string modelPath = path + "/models";
	std::string tempPath = path + "/temp";
	std::string resultPath = path + "/results";
	// window name
	std::string window0 = "Test";

	// display
	std::cout << "Distortion-Free Wide-Angle Portraits on Camera Phones" << std::endl;
	// get image info
	std::cout << "The path of the selected image: ";
	std::string selectedImagePath;
	std::cin >> selectedImagePath;
	std::cout << "The FOV of the image: ";
	double fov = 0;
	std::cin >> fov;
	fov = angle2radian(fov);  // transform angle to radian value
	 std::cout << "The scale value: ";
	std::cin >> scale;
	std::getchar();  // get Enter

	//// debug: directly set
	//// image characters
	//std::string  selectedImagePath = imagePath + "/test4_FOV97.jpg";
	//double fov = angle2radian(97.0);
	//scale = 8;

	// read in origin image
	cv::Mat imOrigin = cv::imread(selectedImagePath);
	if (imOrigin.empty()) {
		std::cout << "Error: Fail to read the origin image." << std::endl;
		return -1;
	}

	// interact to generate a mask or import the mask
	char command;
	std::cout << "Mask generation options (0: import; 1: interaction; Enter: for default 0): ";
	std::cin.get(command);
	if (command == '\n') {
		command = '0';
	}
	else {
		std::getchar();  // get Enter
	}
	if (command == '1') {
		mask = cv::Mat::zeros(imOrigin.rows, imOrigin.cols, CV_8UC1);  // initial mask
		cv::Mat imTemp = imOrigin.clone();
		cv::namedWindow(winGetRegion, cv::WindowFlags::WINDOW_NORMAL);
		cv::imshow(winGetRegion, imOrigin);
		cv::setMouseCallback(winGetRegion, onMouseGetRegion, reinterpret_cast<void*>(&imTemp));
		// draw region a little bigger is better
		while (true) {
			int key = cv::waitKey(5);
			if (key == 13 or key == 27) {
				std::cout << "The number of selected regions: " << k << std::endl;
				if (!cv::imwrite(tempPath + "/mask.png", mask)) {
					std::cout << "Error: Fail to save the mask.";
				}
				else {
					std::cout << "Mask image has been generated and saved." << std::endl;
				}
				cv::destroyWindow(winGetRegion);
				break;
			}
		}
		//// display origin image
		//cv::namedWindow(window0, cv::WindowFlags::WINDOW_NORMAL);
		//cv::imshow(window0, imOrigin);
		//cv::waitKey(0);
	}
	else if (command == '0') {
		// read in mask
		std::cout << "The path of mask image (Enter for default): ";
		std::string temp;
		char flag;
		//char flag;
		std::cin.get(flag);
		if (flag == '\n') {
			temp = tempPath + "/mask.png";  // default path
		}
		else {
			std::string temp2;
			temp.push_back(flag);
			std::cin >> temp2;
			temp += temp2;
		}
		mask = cv::imread(temp, cv::ImreadModes::IMREAD_GRAYSCALE);
		if (mask.data == nullptr) {
			std::cout << "Error: Fail to read the mask image." << std::endl;
			return -1;
		}
		else {
			std::cout << "The number of selected regions: ";
			std::cin >> k;
			std::getchar();
		}
		//// display the mask
		//cv::namedWindow(window0, cv::WindowFlags::WINDOW_NORMAL);
		//cv::imshow(window0, mask);
		//cv::waitKey(0);
	}

	//// debug: import mask directly
	//cv::Mat mask = cv::imread(tempPath + "/mask.png", cv::ImreadModes::IMREAD_GRAYSCALE);
	//k = 5;
	//// check invalid k
	//std::cout << "Checking the validity of k: " << k << std::endl;
	//for (int i = 0; i < mask.cols; ++i) {
	//	for (int j = 0; j < mask.rows; ++j) {
	//		if (mask.at<uchar>(cv::Point(i, j)) > k) {
	//			std::cout << "Exception (" << i << ", " << j << "): " 
	//				<< static_cast<int>(mask.at<uchar>(cv::Point(i, j))) << std::endl;
	//		}
	//	}
	//}

	// obtain image characters
	int w = imOrigin.cols;
	int h = imOrigin.rows;
	int imRows = h;
	int imCols = w;
	// there are three choices for d, determined by the fov for horizontal, vertical or diagonal
	double d1 = std::min<int>(w, h);
	double d2 = std::max<int>(w, h);
	double d3 = sqrt(w * w + h * h) / 2.0;
	double d = d2;
	double f =  d * atan(fov / 2.0);
	double r0 = d / (2 * tan(atan(d / 2 / f) / 2.0));

	// define perspective mesh
	int meshRows = ceil(h / static_cast<double>(scale)) + 1;
	int meshCols = ceil(w / static_cast<double>(scale)) + 1;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> meshPx;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> meshPy;
	meshPx.resize(meshRows, meshCols);
	meshPy.resize(meshRows, meshCols);
	for (int i = 0; i < meshRows; ++i) {
		for (int j = 0; j < meshCols; ++j) {
			meshPx(i, j) = j;
		}
	}
	for (int i = 0; i < meshCols; ++i) {
		for (int j = 0; j < meshRows; ++j) {
			meshPy(j, i) = j;
		}
	}
	
	// calculate stereographic mesh
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> meshUx;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> meshUy;
	meshUx.resize(meshRows, meshCols);
	meshUy.resize(meshRows, meshCols);
	
	int meshRowsCenter = static_cast<int>(meshRows / 2.0);
	int meshColsCenter = static_cast<int>(meshCols / 2.0);
	for (int i = 0; i < meshRows; ++i) {
		for (int j = 0; j < meshCols; ++j) {
			double rP = scale * sqrt(pow(meshPx(i, j) - meshColsCenter, 2) + pow(meshPy(i, j) - meshRowsCenter, 2));
			double rU = r0 * tan(atan(rP / f) / 2.0);
			if (i == meshRowsCenter and j == meshColsCenter) {
				meshUx(i, j) = meshColsCenter;
				meshUy(i, j) = meshRowsCenter;
			}
			else {
				meshUx(i, j) = meshColsCenter + (meshPx(i, j) - meshColsCenter) * rP / rU;
				meshUy(i, j) = meshRowsCenter + (meshPy(i, j) - meshRowsCenter) * rP / rU;
			}
		}
	}

	// draw mesh images and transform photo images
	// generate mesh image from mesh's x and y positions
	cv::Mat imMeshP = getMesh(meshPx, meshPy, true);
	cv::imwrite(tempPath + "/meshP.png", imMeshP);

	// generate mesh image from mesh's x and y positions
	cv::Mat imMeshU = getMesh(meshUx, meshUy, true);
	cv::imwrite(tempPath + "/meshU.png", imMeshU);

	// stretch the image with perspective mesh
	cv::Mat imP = stretch(imOrigin, meshPx, meshPy);
	cv::imwrite(tempPath + "/imP.png", imP);

	// stretch the image with stereographic mesh
	cv::Mat imU = stretch(imOrigin, meshUx, meshUy);
	cv::imwrite(tempPath + "/imU.png", imU);

	// stretch the image with stereographic mesh for the selected region
	cv::Mat imUMask = stretchMask(imOrigin, meshUx, meshUy, mask);
	cv::imwrite(tempPath + "/imUMask.png", imUMask);

	// optimization: local face undistortion
	// initialize logging
	google::InitGoogleLogging("WideAnglePotrait");

	// mask matrix
	Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic> matMask;
	matMask.resize(meshRows, meshCols);
	// sample and initialize the matrix
	for (int i = 0; i < meshRows; ++i) {
		for (int j = 0; j < meshCols; ++j) {
			if (i * scale >= mask.rows or j * scale >= mask.cols) {
				matMask(i, j) = 0;
			}
			else {
				matMask(i, j) = mask.at<uchar>(cv::Point2i(j * scale, i * scale));
			}
		}
	}

	//// weight matrix (not used now)
	//Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> matW;
	//matW.resize(meshRows, meshCols);
	//for (int i = 0; i < meshRows; ++i) {
	//	for (int j = 0; j < meshCols; ++j) {
	//		matW(i, j) = matMask(i, j) > 0;
	//	}
	//}

	//// count the pixels' number of each region
	//std::vector<int> pixelNumber(k, 0);
	//for (int i = 0; i < meshRows; ++i) {
	//	for (int j = 0; j < meshCols; ++j) {
	//		for (int kk = 1; kk < k + 1; ++kk) {
	//			if (matMask(i, j) == kk) {
	//				pixelNumber[kk - 1] += 1;
	//				continue;
	//			}
	//		}
	//		if (matMask(i, j) < 0 or matMask(i, j) > k) {
	//			std::cout << "Exception k value: " << static_cast<int>(matMask(i, j))
	//				<< " in (" << i << ", " << j << ")." << std::endl;
	//		}
	//	}
	//}
	//for (auto it = pixelNumber.begin(); it != pixelNumber.end(); ++it) {
	//	std::cout << *it * scale * scale << " ";
	//}
	//std::cout << std::endl;

	// define and initialize variables
	int q = 4;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matVx, matVy;
	int meshRowsE = meshRows + 2 * q;
	int meshColsE = meshCols + 2 * q;
	matVx.resize(meshRowsE, meshColsE);
	matVy.resize(meshRowsE, meshColsE);
	for (int i = q; i < meshRows + q; ++i) {
		for (int j = q; j < meshCols + q; ++j) {
			int ii = i - q;
			int jj = j - q;
			double deltaVx = 0, deltaVy = 0;
			double norm = 1;  // 1 for non-selected region points
			int kk = matMask(ii, jj);  // current k value
			if (kk > 0) {
				norm = 0;
				// region points should be dealt with
				for (int k = 0; k < meshRows; ++k) {
					for (int l = 0; l < meshCols; ++l) {
						if (matMask(k, l) == kk) {
							double temp = Kh(
								pow(meshPx(k, l) - meshPx(ii, jj), 2)
								+ pow(meshPy(k, l) - meshPy(ii, jj), 2));
							deltaVx += temp * (meshUx(k, l) - meshPx(k, l));
							deltaVy += temp * (meshUy(k, l) - meshPy(k, l));
							norm += temp;
						}
					}
				}
			}
			matVx(i, j) = deltaVx / norm + meshPx(ii, jj);
			matVy(i, j) = deltaVy / norm + meshPy(ii, jj);
		}
	}

	// boundary initialize (initial values also fixed boundary values)
	for (int i = 0; i < q; ++i) {
		for (int j = 0; j < q; ++j) {
			matVx(i, j) = meshPx(0, 0);
			matVy(i, j) = meshPy(0, 0);
		}
		for (int j = q; j < meshCols + q; ++j) {
			matVx(i, j) = meshPx(0, j - q);
			matVy(i, j) = meshPy(0, j - q);
		}
		for (int j = meshCols + q; j < meshCols + q * 2; ++j) {
			matVx(i, j) = meshPx(0, meshCols - 1);
			matVy(i, j) = meshPy(0, meshCols - 1);
		}
	}
	for (int i = meshRows + q; i < meshRows + q * 2; ++i) {
		for (int j = 0; j < q; ++j) {
			matVx(i, j) = meshPx(meshRows - 1, 0);
			matVy(i, j) = meshPy(meshRows - 1, 0);
		}
		for (int j = q; j < meshCols + q; ++j) {
			matVx(i, j) = meshPx(meshRows - 1, j - q);
			matVy(i, j) = meshPy(meshRows - 1, j - q);
		}
		for (int j = meshCols + q; j < meshCols + q * 2; ++j) {
			matVx(i, j) = meshPx(meshRows - 1, meshCols - 1);
			matVy(i, j) = meshPy(meshRows - 1, meshCols - 1);
		}
	}
	for (int j = 0; j < q; ++j) {
		for (int i = q; i < meshRows + q; ++i) {
			matVx(i, j) = meshPx(i - q, 0);
			matVy(i, j) = meshPy(i - q, 0);
		}
	}
	for (int j = q + meshCols; j < meshCols + 2 * q; ++j) {
		for (int i = q; i < meshRows + q; ++i) {
			matVx(i, j) = meshPx(i - q, meshCols - 1);
			matVy(i, j) = meshPy(i - q, meshCols - 1);
		}
	}

	//// debug: use the saved matrix as initial value
	//matVx = extendPadding<double>(readMatrix<double>(tempPath + "/matxOptima.txt"), q);
	//matVy = extendPadding<double>(readMatrix<double>(tempPath + "/matyOptima.txt"), q);

	// Sk and tk
	// Sk is composed of ak and bk
	std::vector<double> as, bs;
	as.resize(k);
	bs.resize(k);
	for (int i = 0; i < k; ++i) {
		as[i] = 1;
		bs[i] = 0;
	}
	// tk is composed of x value and y value
	std::vector<double> txs, tys;
	txs.resize(k);
	tys.resize(k);
	for (int i = 0; i < k; ++i) {
		txs[i] = 0;
		tys[i] = 0;
	}

	// extend the size of defined matrix to fit padding demands
	//auto matWE = extendPadding<bool>(matW, q);
	auto matMaskE = extendPadding<uchar>(matMask, q);
	auto meshPxE = extendPadding<double>(meshPx, q);
	auto meshPyE = extendPadding<double>(meshPy, q);
	auto meshUxE = extendPadding<double>(meshUx, q);
	auto meshUyE = extendPadding<double>(meshUy, q);

	// cut the padding of the initial solution mesh
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matVxInitFinal, matVyInitFinal;
	matVxInitFinal = cutPadding<double>(matVx, q);
	matVyInitFinal = cutPadding<double>(matVy, q);

	// generate mesh image from mesh's x and y positions
	cv::Mat imMeshInit = getMesh(matVxInitFinal, matVyInitFinal, true);
	cv::imwrite(tempPath + "/meshInit.png", imMeshInit);

	// stretch the image with stereographic mesh
	cv::Mat imInit = stretch(imOrigin, matVxInitFinal, matVyInitFinal);
	cv::imwrite(tempPath + "/imInit.png", imInit);

	// define problem
	// we can also set different weights through ConditionedCostFunction
	// here we integrate weights in the cost function
	ceres::Problem problem;

	// m function params
	double eps = 1e-10;
	double rb = 2 * tan(angle2radian(50)) * f / (log(99) - log(eps));
	double ra = log(99) * rb;
	double pCenterx = meshPx(meshRowsCenter, meshColsCenter);
	double pCentery = meshPy(meshRowsCenter, meshColsCenter);

	// lambda(Sk) parameters
	double ws = 2000.0;
	double st = 1.0;

	// Ef part1
	// consider padding
	for (int i = 0; i < meshRowsE; ++i) {
		for (int j = 0; j < meshColsE; ++j) {
			int kIndex = matMaskE(i, j) - 1;  // current k's index
			// if kIndex < 0, i.e. k = 0, not our selected regions
			if (kIndex < 0) {
				problem.AddParameterBlock(&matVx(i, j), 1);
				problem.AddParameterBlock(&matVy(i, j), 1);
			}
			else if (kIndex >= 0 and kIndex <= k - 1) {
				problem.AddResidualBlock(
					new ceres::AutoDiffCostFunction<costEf1, 2, 1, 1, 1, 1, 1, 1>(
						new costEf1(m(sqrt(pow(meshPxE(i, j) - pCenterx, 2) + pow(meshPyE(i, j) - pCentery, 2)), ra, rb),
							meshUxE(i, j), meshUyE(i, j))
						),
					nullptr,
					&matVx(i, j), &matVy(i, j), &as[kIndex], &bs[kIndex], &txs[kIndex], &tys[kIndex]);
			}
			else {
				std::cerr << "Error: The value of k is out of the bound -> " << kIndex + 1 << std::endl;
			}
		}
	}

	//Ef part2
	for (int i = 0; i < k; ++i) {
		problem.AddResidualBlock(
			new ceres::AutoDiffCostFunction<costEf2, 1, 1>(
				new costEf2(ws, st)
				),
			nullptr,
			&as[i]
		);
	}

	// Eb and Er
	// consider padding
	for (int i = 0; i < meshRowsE; ++i) {
		for (int j = 0; j < meshColsE; ++j) {
			int ii = 0;
			int jj = 0;
			// on the corner
			if (i == 0 and j == 0) {
				ii = i;
				jj = j + 1;
				// only lambdaR (only Er)
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				// both lambdaR and lambdaB (both Eb and Er)
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVy(ii, jj), &matVy(i, j));
				ii = i + 1;
				jj = j;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVy(ii, jj), &matVy(i, j));
			}
			else if (i == 0 and j == meshColsE - 1) {
				ii = i;
				jj = j - 1;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVy(ii, jj), &matVy(i, j));
				ii = i + 1;
				jj = j;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVy(ii, jj), &matVy(i, j));
			}
			else if (i == meshRowsE - 1 and j == meshColsE - 1) {
				ii = i;
				jj = j - 1;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVy(ii, jj), &matVy(i, j));
				ii = i - 1;
				jj = j;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVy(ii, jj), &matVy(i, j));
			}
			else if (i == meshRowsE - 1 and j == 0) {
				ii = i;
				jj = j + 1;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVy(ii, jj), &matVy(i, j));
				ii = i - 1;
				jj = j;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVy(ii, jj), &matVy(i, j));
			}
			// on the side but not on the corner
			else if (i == 0) {
				ii = i;
				jj = j - 1;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVy(ii, jj), &matVy(i, j));
				jj = j + 1;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVy(ii, jj), &matVy(i, j));
				ii = i + 1;
				jj = j;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVy(ii, jj), &matVy(i, j));
			}
			else if (i == meshRowsE - 1) {
				ii = i;
				jj = j - 1;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVy(ii, jj), &matVy(i, j));
				jj = j + 1;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVy(ii, jj), &matVy(i, j));
				ii = i - 1;
				jj = j;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVy(ii, jj), &matVy(i, j));
			}
			else if (j == 0) {
				ii = i;
				jj = j + 1;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVy(ii, jj), &matVy(i, j));
				ii = i - 1;
				jj = j;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVy(ii, jj), &matVy(i, j));
				ii = i + 1;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVy(ii, jj), &matVy(i, j));
			}
			else if (j == meshColsE - 1) {
				ii = i;
				jj = j - 1;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVy(ii, jj), &matVy(i, j));
				ii = i - 1;
				jj = j;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVy(ii, jj), &matVy(i, j));
				ii = i + 1;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVy(ii, jj), &matVy(i, j));
			}
			// on casual position but not special
			else {
				ii = i;
				jj = j - 1;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVy(ii, jj), &matVy(i, j));
				jj = j + 1;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVy(ii, jj), &matVy(i, j));
				ii = i - 1;
				jj = j;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVy(ii, jj), &matVy(i, j));
				ii = i + 1;
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb()),
					nullptr, &matVx(ii, jj), &matVx(i, j));
				problem.AddResidualBlock(new ceres::AutoDiffCostFunction<costEb, 1, 1, 1>(new costEb(0)),
					nullptr, &matVy(ii, jj), &matVy(i, j));
			}
		}
	}

	// Ea
	// And boundary conditions
	int i = 0, j = 0;
	double epsilonBound = 1e-10;  // mask
    // El
	j = 0;
	for (int i = 0; i < meshRowsE; ++i) {
		problem.AddResidualBlock(
			new ceres::AutoDiffCostFunction<costEaG, 1, 1>(
				new costEaG(0.0)
				),
			nullptr,
			&matVx(i, j)
		);
		// note that the second parameter of function is index but size
		problem.SetParameterUpperBound(&matVx(i, j), 0, matVx(i, j) + epsilonBound);
		problem.SetParameterLowerBound(&matVx(i, j), 0, matVx(i, j) - epsilonBound);
	}
	// Er
	j = meshColsE - 1;
	for (int i = 0; i < meshRowsE; ++i) {
		problem.AddResidualBlock(
			new ceres::AutoDiffCostFunction<costEaL, 1, 1>(
				new costEaL(static_cast<double>(meshColsE - 1))
				),
			nullptr,
			&matVx(i, j)
		);
		problem.SetParameterUpperBound(&matVx(i, j), 0, matVx(i, j) + epsilonBound);
		problem.SetParameterLowerBound(&matVx(i, j), 0, matVx(i, j) - epsilonBound);
	}
	// Et
	i = 0;
	for (int j = 0; j < meshColsE; ++j) {
		problem.AddResidualBlock(
			new ceres::AutoDiffCostFunction<costEaG, 1, 1>(
				new costEaG(0.0)
				),
			nullptr,
			&matVy(i, j)
		);
		problem.SetParameterUpperBound(&matVy(i, j), 0, matVy(i, j) + epsilonBound);
		problem.SetParameterLowerBound(&matVy(i, j), 0, matVy(i, j) - epsilonBound);
	}
	// Eb
	i = meshRowsE - 1;
	for (int j = 0; j < meshColsE; ++j) {
		problem.AddResidualBlock(
			new ceres::AutoDiffCostFunction<costEaL, 1, 1>(
				new costEaL(static_cast<double>(meshRowsE - 1))
				),
			nullptr,
			&matVy(i, j)
		);
		problem.SetParameterUpperBound(&matVy(i, j), 0, matVy(i, j) + epsilonBound);
		problem.SetParameterLowerBound(&matVy(i, j), 0, matVy(i, j) - epsilonBound);
	}

	// options configure
	ceres::Solver::Options options;
	options.minimizer_progress_to_stdout = true;
	options.minimizer_type = ceres::TRUST_REGION;
	options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.initial_trust_region_radius = 1e6;  // mask
	//options.use_explicit_schur_complement = true;
	//options.use_inner_iterations = true;
	options.num_threads = 128;
	//options.max_num_iterations = 1e5;
	//options.function_tolerance = 1e-6;

	// solve
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

	// obtain optimization result
	// without normalization in the paper for on need here -> mask
	// cut the padding of the optimized mesh
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matVxFinal, matVyFinal;
	matVxFinal = cutPadding<double>(matVx, q);
	matVyFinal = cutPadding<double>(matVy, q);
	
	// generate mesh image from mesh's x and y positions
	cv::Mat imMeshOptima = getMesh(matVxFinal, matVyFinal, true);
	cv::imwrite(tempPath + "/meshOptima.png", imMeshOptima);

	// stretch the image with stereographic mesh
	cv::Mat imOptima = stretch(imOrigin, matVxFinal, matVyFinal);
	cv::imwrite(tempPath + "/imOptima.png", imOptima);

	// output parameters
	std::cout << "Parameter a[k]: ";
	for (int i = 0; i < as.size(); ++i) {
		std::cout << as[i] << ", ";
	}
	std::cout << std::endl;
	std::cout << "Parameter b[k]: ";
	for (int i = 0; i < as.size(); ++i) {
		std::cout << bs[i] << ", ";
	}
	std::cout << std::endl;
	std::cout << "Parameter t[k]: ";
	for (int i = 0; i < as.size(); ++i) {
		std::cout << txs[i] << ":" << tys[i] << ", ";
	}
	std::cout << std::endl;

	// save the optimized mesh matrix
	writeMatrix<double>(tempPath + "/matxOptima.txt", matVx);
	writeMatrix<double>(tempPath + "/matyOptima.txt", matVy);

	system("pause");

	return 0;
}


// m function
double m(double r, double ra, double rb)
{
	return 1.0 / (1 + exp((ra - r) / rb));
}


// Kh function
double Kh(double x)
{
	double h = 2.37;
	return exp(-x / (2 * h * h));
}


// angle to radian
double angle2radian(double angle)
{
	return angle / 180.0 * CV_PI;
}

// mouse event register function: get click point's position and value
void onMouseClick(int event, int x, int y, int flags, void* param)
{
	cv::Mat* image = reinterpret_cast<cv::Mat*>(param);

	switch (event) {
	case cv::EVENT_LBUTTONDOWN:
		if (image->channels() == 1) {
			std::cout << "At (" << x << ", " << y << "): " << static_cast<int>(image->at<uchar>(cv::Point(x, y))) << std::endl;
		}
		else if (image->channels() == 3) {
			std::cout << "At (" << x << ", " << y << "): " << image->at<cv::Vec3b>(cv::Point(x, y)) << std::endl;
		}
		else {
			std::cout << "Not support current image with channel number except 1 and 3." << std::endl;
		}
		break;
	}
}


// mouse event register function: interact to get a region
void onMouseGetRegion(int event, int x, int y, int flags, void* param)
{
	// params
	cv::Scalar lineColor(255, 144, 30);
	cv::Scalar regionColor(250, 206, 135);
	cv::Mat* image = reinterpret_cast<cv::Mat*>(param);
	
	if (event == cv::EVENT_LBUTTONUP || !(flags & cv::EVENT_FLAG_LBUTTON)) {
		prePoint = cv::Point(-1, -1);
	}
	else if (event == cv::EVENT_LBUTTONDOWN) {
		prePoint = cv::Point2i(x, y);
	}
	else if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON)) {
		cv::Point2i point(x, y);
		cv::line(*image, prePoint, point, lineColor, 3, 8, 0);
		cv::line(mask, prePoint, point, cv::Scalar(k + 1), 1, 8, 0);
		prePoint = point;
		cv::imshow(winGetRegion, *image);
	}

	// fill the region
	if (event == cv::EVENT_RBUTTONUP) {
		cv::floodFill(*image, cv::Point2i(x, y), regionColor);
		cv::floodFill(mask, cv::Point2i(x, y), cv::Scalar(k + 1));  // k + 1 is the mask value for different regions
		cv::Mat temp(image->rows, image->cols, CV_8UC3, regionColor);
		temp.copyTo(*image, mask);
		cv::imshow(winGetRegion, *image);
		std::cout << "Current region's k: " << k + 1 << std::endl;
		k += 1;  // update the number of selected regions
	}
}


// get mesh image max through meshx and meshy
cv::Mat getMesh(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> meshx,
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> meshy,
	bool withLine)
{
	cv::Mat imMesh = cv::Mat((meshx.rows() - 1) * scale, (meshx.cols() - 1) * scale, CV_8UC1, cv::Scalar(0));

	// draw points
	for (int i = 0; i < meshx.rows(); ++i) {
		for (int j = 0; j < meshy.cols(); ++j) {
			// draw points
			cv::circle(imMesh, cv::Point2i(meshx(i, j) * scale, meshy(i, j) * scale), 1, cv::Scalar(255), -1);
			if (withLine) {
				// draw lines
				cv::line(imMesh,
					cv::Point2i((meshx(i, j)) * scale, (meshy(i, j)) * scale),
					cv::Point2i((meshx(i, j) - 1) * scale, (meshy(i, j)) * scale),
					cv::Scalar(255),
					1);
				cv::line(imMesh,
					cv::Point2i((meshx(i, j)) * scale, (meshy(i, j)) * scale),
					cv::Point2i((meshx(i, j) + 1) * scale, (meshy(i, j)) * scale),
					cv::Scalar(255),
					1);
				cv::line(imMesh,
					cv::Point2i((meshx(i, j)) * scale, (meshy(i, j)) * scale),
					cv::Point2i((meshx(i, j)) * scale, (meshy(i, j) - 1) * scale),
					cv::Scalar(255),
					1);
				cv::line(imMesh,
					cv::Point2i((meshx(i, j)) * scale, (meshy(i, j)) * scale),
					cv::Point2i((meshx(i, j)) * scale, (meshy(i, j) + 1) * scale),
					cv::Scalar(255),
					1);
			}
		}
	}

	return imMesh;
}


// stretch the image with mesh (from meshP to mesh)
cv::Mat stretch(cv::Mat imOrigin,
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> meshx,
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> meshy)
{
	cv::Mat imResult;
	cv::Mat mapx = cv::Mat::zeros(imOrigin.rows, imOrigin.cols, CV_32FC1);
	cv::Mat mapy = cv::Mat::zeros(imOrigin.rows, imOrigin.cols, CV_32FC1);
	
	double s = scale;
	double r = s * sqrt(2);
	// protect version
	//for (int i = 0; i < imOrigin.cols; ++i) {
	//	for (int j = 0; j < imOrigin.rows; ++j) {
	//		if (i % scale == 0 and j % scale == 0) {
	//			// at grid point
	//			mapx.at<float>(cv::Point(i, j)) = scale * meshx(static_cast<int>(j / s), static_cast<int>(i / s));
	//			mapy.at<float>(cv::Point(i, j)) = scale * meshy(static_cast<int>(j / s), static_cast<int>(i / s));
	//		}
	//		else if (i % scale == 0 and j % scale != 0) {
	//			// at grid line on y axis
	//			int m = static_cast<int>(i / s);
	//			int n = static_cast<int>(floor(j / s));
	//			double alpha2 = (j - n * scale) / s;
	//			double alpha1 = 1 - alpha2;
	//			if (n == meshx.rows() - 1) {
	//				mapx.at<float>(cv::Point(i, j)) = scale * meshx(n, m);
	//				mapy.at<float>(cv::Point(i, j)) = scale * meshy(n, m);
	//			}
	//			else {
	//				mapx.at<float>(cv::Point(i, j)) = scale * (meshx(n, m) * alpha1 + meshx(n + 1, m) * alpha2);
	//				mapy.at<float>(cv::Point(i, j)) = scale * (meshy(n, m) * alpha1 + meshy(n + 1, m) * alpha2);
	//			}
	//		}
	//		else if (i % scale != 0 and j % scale == 0) {
	//			// at grid line on x axis
	//			int m = static_cast<int>(floor(i / s));
	//			int n = static_cast<int>(j / s);
	//			double alpha2 = (i - m * scale) / s;
	//			double alpha1 = 1 - alpha2;
	//			if (m == meshx.cols() - 1) {
	//				mapx.at<float>(cv::Point(i, j)) = scale * meshx(n, m);
	//				mapy.at<float>(cv::Point(i, j)) = scale * meshy(n, m);
	//			}
	//			else {
	//				mapx.at<float>(cv::Point(i, j)) = scale * (meshx(n, m) * alpha1 + meshx(n, m + 1) * alpha2);
	//				mapy.at<float>(cv::Point(i, j)) = scale * (meshy(n, m) * alpha1 + meshy(n, m + 1) * alpha2);
	//			}
	//		}
	//		else {
	//			// inside grid
	//			int m = static_cast<int>(floor(i / s));
	//			int n = static_cast<int>(floor(j / s));
	//			double alpha2 = (i - m * scale) / s;
	//			double alpha1 = 1 - alpha2;
	//			double belta2 = (j - n * scale) / s;
	//			double belta1 = 1 - belta2;
	//			if (n == meshx.rows() - 1 and m == meshx.cols() - 1) {
	//				mapx.at<float>(cv::Point(i, j)) = scale * meshx(n, m);
	//			}
	//			else if (n == meshx.rows() - 1) {
	//				mapx.at<float>(cv::Point(i, j)) = scale * (meshx(n, m) * alpha1 + meshx(n, m + 1) * alpha2);
	//				mapy.at<float>(cv::Point(i, j)) = scale * ((meshy(n, m) + meshy(n, m + 1)) / 2.0);
	//			}
	//			else if (m == meshx.cols() - 1) {
	//				mapx.at<float>(cv::Point(i, j)) = scale * ((meshx(n, m) + meshx(n + 1, m)) / 2.0);
	//				mapy.at<float>(cv::Point(i, j)) = scale * (meshy(n, m) * belta1 + meshy(n + 1, m) * belta2);
	//			}
	//			else {
	//				mapx.at<float>(cv::Point(i, j)) = scale * ((meshx(n, m) + meshx(n + 1, m)) * alpha1 / 2 +
	//					(meshx(n, m + 1) + meshx(n + 1, m + 1)) * alpha2 / 2);
	//				mapy.at<float>(cv::Point(i, j)) = scale * ((meshy(n, m) + meshy(n, m + 1)) * belta1 / 2 +
	//					(meshy(n + 1, m) + meshy(n + 1, m + 1)) * belta2 / 2);
	//			}
	//		}
	//	}
	//}
	// no protect version but faster (also speed up with pointers)

	double u1x = 0, u1y = 0, u2x = 0, u2y = 0, u3x = 0, u3y = 0, u4x = 0, u4y = 0;
	for (int j = 0; j != imOrigin.rows; ++j) {
		float* p_rowx = mapx.ptr<float>(j);
		float* p_rowy = mapy.ptr<float>(j);
		for (int i = 0; i != imOrigin.cols; ++i) {
			if (i % scale == 0 and j % scale == 0) {
				// at grid point
				p_rowx[i] = scale * meshx(static_cast<int>(j / s), static_cast<int>(i / s));
				p_rowy[i] = scale * meshy(static_cast<int>(j / s), static_cast<int>(i / s));
			}
			else if (i % scale == 0 and j % scale != 0) {
				// at grid line on y axis
				int m = static_cast<int>(i / s);
				int n = static_cast<int>(floor(j / s));
				double alpha2 = (j - n * scale) / s;
				double alpha1 = 1 - alpha2;
				p_rowx[i] = scale * (meshx(n, m) * alpha1 + meshx(n + 1, m) * alpha2);
				p_rowy[i] = scale * (meshy(n, m) * alpha1 + meshy(n + 1, m) * alpha2);
			}
			else if (i % scale != 0 and j % scale == 0) {
				// at grid line on x axis
				int m = static_cast<int>(floor(i / s));
				int n = static_cast<int>(j / s);
				double alpha2 = (i - m * scale) / s;
				double alpha1 = 1 - alpha2;
				p_rowx[i] = scale * (meshx(n, m) * alpha1 + meshx(n, m + 1) * alpha2);
				p_rowy[i] = scale * (meshy(n, m) * alpha1 + meshy(n, m + 1) * alpha2);
			}
			else {
				// inside grid
				int m = static_cast<int>(floor(i / s));
				int n = static_cast<int>(floor(j / s));
				//// x, y independent coefficients version
				//double alpha2 = (i - m * scale) / s;
				//double alpha1 = 1 - alpha2;
				//double belta2 = (j - n * scale) / s;
				//double belta1 = 1 - belta2;
				//p_rowx[i] = scale * ((meshx(n, m) + meshx(n + 1, m)) * alpha1 / 2 +
				//	(meshx(n, m + 1) + meshx(n + 1, m + 1)) * alpha2 / 2);
				//p_rowy[i] = scale * ((meshy(n, m) + meshy(n, m + 1)) * belta1 / 2 +
				//	(meshy(n + 1, m) + meshy(n + 1, m + 1)) * belta2 / 2);
				// strict version
				int ir = i % scale;
				int jr = j % scale;
				u1x = meshx(n, m) + jr / s * (meshx(n + 1, m) - meshx(n, m));
				u1y = meshy(n, m) + jr / s * (meshy(n + 1, m) - meshy(n, m));
				u2x = meshx(n, m) + ir / s * (meshx(n, m + 1) - meshx(n, m));
				u2y = meshy(n, m) + ir / s * (meshy(n, m + 1) - meshy(n, m));
				u3x = meshx(n, m + 1) + jr / s * (meshx(n + 1, m + 1) - meshx(n, m + 1));
				u3y = meshy(n, m + 1) + jr / s * (meshy(n + 1, m + 1) - meshy(n, m + 1));
				u4x = meshx(n + 1, m) + ir / s * (meshx(n + 1, m + 1) - meshx(n + 1, m));
				u4y = meshy(n + 1, m) + ir / s * (meshy(n + 1, m + 1) - meshy(n + 1, m));
				p_rowx[i] = scale * (-(-u2x * u3x * u1y + u3x * u4x * u1y + u1x * u4x * u2y - u3x * u4x * u2y
					+ u1x * u2x * u3y - u1x * u4x * u3y + u2x * u3x * u4y - u1x * u2x * u4y) /
					(u2x * u1y - u4x * u1y - u1x * u2y + u3x * u2y - u2x * u3y + u4x * u3y + u1x * u4y - u3x * u4y));
				p_rowy[i] = scale * (((-u3x * u1y + u1x * u3y) * (u2y - u4y) - (u1y - u3y) * (-u4x * u2y + u2x * u4y)) /
					((-u2x + u4x) * (u1y - u3y) - (-u1x + u3x) * (u2y - u4y)));
			}
		}
	}

	cv::remap(imOrigin, imResult, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

	return imResult;
}


// stretch the image with mesh just for the selected region
cv::Mat stretchMask(cv::Mat imOrigin,
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> meshx,
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> meshy,
	cv::Mat mask)
{
	cv::Mat imResult;
	cv::Mat mapx = cv::Mat::zeros(imOrigin.rows, imOrigin.cols, CV_32FC1);
	cv::Mat mapy = cv::Mat::zeros(imOrigin.rows, imOrigin.cols, CV_32FC1);

	double s = scale;
	double u1x = 0, u1y = 0, u2x = 0, u2y = 0, u3x = 0, u3y = 0, u4x = 0, u4y = 0;
	for (int j = 0; j != imOrigin.rows; ++j) {
		float* p_rowx = mapx.ptr<float>(j);
		float* p_rowy = mapy.ptr<float>(j);
		for (int i = 0; i != imOrigin.cols; ++i) {
			// check if the pixel is selected
			if (mask.at<uchar>(cv::Point2i(i, j)) == 0) {
				// not selected
				p_rowx[i] = i;
				p_rowy[i] = j;
			}
			else {
				// selected
				if (i % scale == 0 and j % scale == 0) {
					// at grid point
					p_rowx[i] = scale * meshx(static_cast<int>(j / s), static_cast<int>(i / s));
					p_rowy[i] = scale * meshy(static_cast<int>(j / s), static_cast<int>(i / s));
				}
				else if (i % scale == 0 and j % scale != 0) {
					// at grid line on y axis
					int m = static_cast<int>(i / s);
					int n = static_cast<int>(floor(j / s));
					double alpha2 = (j - n * scale) / s;
					double alpha1 = 1 - alpha2;
					p_rowx[i] = scale * (meshx(n, m) * alpha1 + meshx(n + 1, m) * alpha2);
					p_rowy[i] = scale * (meshy(n, m) * alpha1 + meshy(n + 1, m) * alpha2);
				}
				else if (i % scale != 0 and j % scale == 0) {
					// at grid line on x axis
					int m = static_cast<int>(floor(i / s));
					int n = static_cast<int>(j / s);
					double alpha2 = (i - m * scale) / s;
					double alpha1 = 1 - alpha2;
					p_rowx[i] = scale * (meshx(n, m) * alpha1 + meshx(n, m + 1) * alpha2);
					p_rowy[i] = scale * (meshy(n, m) * alpha1 + meshy(n, m + 1) * alpha2);
				}
				else {
					// inside grid
					int m = static_cast<int>(floor(i / s));
					int n = static_cast<int>(floor(j / s));
					// strict version
					int ir = i % scale;
					int jr = j % scale;
					u1x = meshx(n, m) + jr / s * (meshx(n + 1, m) - meshx(n, m));
					u1y = meshy(n, m) + jr / s * (meshy(n + 1, m) - meshy(n, m));
					u2x = meshx(n, m) + ir / s * (meshx(n, m + 1) - meshx(n, m));
					u2y = meshy(n, m) + ir / s * (meshy(n, m + 1) - meshy(n, m));
					u3x = meshx(n, m + 1) + jr / s * (meshx(n + 1, m + 1) - meshx(n, m + 1));
					u3y = meshy(n, m + 1) + jr / s * (meshy(n + 1, m + 1) - meshy(n, m + 1));
					u4x = meshx(n + 1, m) + ir / s * (meshx(n + 1, m + 1) - meshx(n + 1, m));
					u4y = meshy(n + 1, m) + ir / s * (meshy(n + 1, m + 1) - meshy(n + 1, m));
					p_rowx[i] = scale * (-(-u2x * u3x * u1y + u3x * u4x * u1y + u1x * u4x * u2y - u3x * u4x * u2y
						+ u1x * u2x * u3y - u1x * u4x * u3y + u2x * u3x * u4y - u1x * u2x * u4y) /
						(u2x * u1y - u4x * u1y - u1x * u2y + u3x * u2y - u2x * u3y + u4x * u3y + u1x * u4y - u3x * u4y));
					p_rowy[i] = scale * (((-u3x * u1y + u1x * u3y) * (u2y - u4y) - (u1y - u3y) * (-u4x * u2y + u2x * u4y)) /
						((-u2x + u4x) * (u1y - u3y) - (-u1x + u3x) * (u2y - u4y)));
				}
			}
		}
	}

	cv::remap(imOrigin, imResult, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

	return imResult;
}


// cut the matrix's surrounding padding
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> cutPadding(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat,
	int padding)
{
	return mat.block(padding, padding, mat.rows() - 2 * padding, mat.cols() - 2 * padding);
}


// extend the padding of one matrix
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> extendPadding(
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat, 
	int padding)
{
	int q = padding;
	int meshCols = mat.cols();
	int meshRows = mat.rows();
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat2;
	mat2.resize(meshRows + 2 * q, meshCols + 2 * q);
	mat2.block(q, q, meshRows, meshCols) = mat;

	// fill the padding (boundary initialization) (initial values also fixed boundary values)
	for (int i = 0; i < q; ++i) {
		for (int j = 0; j < q; ++j) {
			mat2(i, j) = mat(0, 0);
		}
		for (int j = q; j < meshCols + q; ++j) {
			mat2(i, j) = mat(0, j - q);
		}
		for (int j = meshCols + q; j < meshCols + q * 2; ++j) {
			mat2(i, j) = mat(0, meshCols - 1);
		}
	}
	for (int i = meshRows + q; i < meshRows + q * 2; ++i) {
		for (int j = 0; j < q; ++j) {
			mat2(i, j) = mat(meshRows - 1, 0);
		}
		for (int j = q; j < meshCols + q; ++j) {
			mat2(i, j) = mat(meshRows - 1, j - q);
		}
		for (int j = meshCols + q; j < meshCols + q * 2; ++j) {
			mat2(i, j) = mat(meshRows - 1, meshCols - 1);
		}
	}
	for (int j = 0; j < q; ++j) {
		for (int i = q; i < meshRows + q; ++i) {
			mat2(i, j) = mat(i - q, 0);
		}
	}
	for (int j = q + meshCols; j < meshCols + 2 * q; ++j) {
		for (int i = q; i < meshRows + q; ++i) {
			mat2(i, j) = mat(i - q, meshCols - 1);
		}
	}

	return mat2;
}


// matrix write to specified file
template <typename T>
void writeMatrix(std::string filePath, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat)
{
	std::ofstream fout;
	fout.open(filePath, std::ios::out);
	fout << mat.rows() << " " << mat.cols() << std::endl;
	for (int i = 0; i < mat.rows(); ++i) {
		for (int j = 0; j < mat.cols() - 1; ++j) {
			fout << mat(i, j) << " ";
		}
		fout << mat(i, mat.cols() - 1) << std::endl;
	}
	fout.close();
}


// Matrix read in from specified file
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> readMatrix(std::string filePath)
{
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat;
	std::ifstream fin;
	fin.open(filePath, std::ios::in);
	int rows = 0;
	fin >> rows;
	int cols = 0;
	fin >> cols;
	mat.resize(rows, cols);
	T value;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			fin >> value;
			mat(i, j) = value;
		}
	}

	fin.close();

	return mat;
}
