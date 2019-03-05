// Annt.cpp : Questo file contiene la funzione 'main', in cui inizia e termina l'esecuzione del programma.
//

#include "pch.h"
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <vector>
#include <map>
#include "Debug/include/ANNT.hpp"

using namespace std;
using namespace ANNT;
using namespace ANNT::Neuro;
using namespace ANNT::Neuro::Training;

static const string BASE = "../../dataset/";

bool LoadData(vector<fvector_t>& attributes, string fileName, int expectedFields)
{
	bool  ret = false;

	string path = "";
	path.append(BASE).append(fileName);

	FILE* file;
	errno_t err;

	if (err = fopen_s(&file, path.c_str(), "r") == 0)
	{
		char buff[256];

		while (fgets(buff, 256, file) != nullptr)
		{
			size_t len = strlen(buff);

			while ((len > 0) && (isspace(buff[len - 1])))
			{
				buff[--len] = '\0';
			}

			if (len != 0)
			{
				float attr1, attr2, attr3, attr4, attr5, attr6;

				if ((sscanf_s(buff, "%f;%f;%f", &attr1, &attr2, &attr3) == expectedFields) ||
					(sscanf_s(buff, "%f;%f;%f;%f;%f;%f", &attr1, &attr2, &attr3, &attr4, &attr5, &attr6) == expectedFields))
				{
					if (expectedFields == 3) {
						attributes.push_back(fvector_t({ attr1, attr2, attr3 }));
					}
					else if (expectedFields == 6) {
						attributes.push_back(fvector_t({ attr1, attr2, attr3, attr4, attr5, attr6 }));
					}
				}
			}
		}

		fclose(file);
		ret = true;
	}

	return ret;
}

int main(int argc, char** argv)
{
    std::cout << "Hello World!\n"; 

	vector<fvector_t> xTrain;
	vector<fvector_t> yTrain;
	vector<fvector_t> xTest;

	if (!LoadData(xTrain, "xtrain.csv", 6))
	{
		printf("Failed loading X Training data \n\n");
		return -1;
	}

	if (!LoadData(yTrain, "ytrain.csv", 3))
	{
		printf("Failed loading Y Training data \n\n");
		return -1;
	}

	if (!LoadData(xTest, "xtest.csv", 6))
	{
		printf("Failed loading X Test data \n\n");
		return -1;
	}

	printf("Loaded %zu X Training entries \n\n", xTrain.size());
	printf("Loaded %zu Y Training entries \n\n", yTrain.size());
	printf("Loaded %zu X Test entries \n\n", xTest.size());

	// make sure we have expected number of samples
	int expectedEntries = 100;
	if (xTrain.size() != expectedEntries || yTrain.size() != expectedEntries || xTrain.size() != yTrain.size())
	{
		printf("The data set is expected to provide 100 samples \n\n");
		return -2;
	}

	
	// split the data set into two: training (120 samples) and test (30 samples)
	/*vector<fvector_t> testAttributes = ExtractTestSamples(trainAttributes);
	uvector_t         testLabels = ExtractTestSamples(trainLabels);

	printf("Using %zu samples for training and %zu samples for test \n\n", trainAttributes.size(), testAttributes.size());
	*/

	// perform one hot encoding of train/test labels
	/*vector<fvector_t> encodedTrainLabels = XDataEncodingTools::OneHotEncoding(yTrain, 3);
	vector<fvector_t> encodedTestLabels = XDataEncodingTools::OneHotEncoding(testLabels, 3);*/

	// prepare a 3 layer ANN
	shared_ptr<XNeuralNetwork> net = make_shared<XNeuralNetwork>();

	net->AddLayer(make_shared<XFullyConnectedLayer>(6, 4));
	net->AddLayer(make_shared<XSigmoidActivation>());
	net->AddLayer(make_shared<XFullyConnectedLayer>(4, 4));
	net->AddLayer(make_shared<XSigmoidActivation>());
	net->AddLayer(make_shared<XFullyConnectedLayer>(4, 3));
	net->AddLayer(make_shared<XSigmoidActivation>());

	// create training context with Nesterov optimizer and Cross Entropy cost function
	shared_ptr<XNetworkTraining> netTraining = make_shared<XNetworkTraining>(net,
		make_shared<XNesterovMomentumOptimizer>(0.01f),
		make_shared<XCrossEntropyCost>());

	// using the helper for training ANN to do classification
	XClassificationTrainingHelper trainingHelper(netTraining, argc, argv);
	//trainingHelper.SetTestSamples(testAttributes, encodedTestLabels, testLabels);

	// 40 epochs, 10 samples in batch
	trainingHelper.RunTraining(40, 10, xTrain, yTrain, trainLabels);
	
	return 0;
}

// Per eseguire il programma: CTRL+F5 oppure Debug > Avvia senza eseguire debug
// Per eseguire il debug del programma: F5 oppure Debug > Avvia debug

// Suggerimenti per iniziare: 
//   1. Usare la finestra Esplora soluzioni per aggiungere/gestire i file
//   2. Usare la finestra Team Explorer per connettersi al controllo del codice sorgente
//   3. Usare la finestra di output per visualizzare l'output di compilazione e altri messaggi
//   4. Usare la finestra Elenco errori per visualizzare gli errori
//   5. Passare a Progetto > Aggiungi nuovo elemento per creare nuovi file di codice oppure a Progetto > Aggiungi elemento esistente per aggiungere file di codice esistenti al progetto
//   6. Per aprire di nuovo questo progetto in futuro, passare a File > Apri > Progetto e selezionare il file con estensione sln
