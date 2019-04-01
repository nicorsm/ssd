// Annt.cpp : Questo file contiene la funzione 'main', in cui inizia e termina l'esecuzione del programma.
//

#include "pch.h"
#include <iostream>
#include <string>
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
static const string FILE_NAME = "mlp.annt";

bool LoadData(vector<fvector_t>& attributes, vector<fvector_t>& labels, string fileName)
{
	bool  ret = false;
	int expectedFields = 6 + 3;

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
				float label1, label2, label3;

				if ((sscanf_s(buff, "%f,%f,%f,%f,%f,%f,%f,%f,%f", &attr1, &attr2, &attr3, &attr4, &attr5, &attr6, &label1, &label2, &label3) == expectedFields))
				{
					attributes.push_back(fvector_t({ attr1, attr2, attr3, attr4, attr5, attr6 }));
					labels.push_back(fvector_t({ label1, label2, label3 }));
				}
			}
		}

		fclose(file);
		ret = true;
	}

	return ret;
}

int train(int argc, char** argv) {

	vector<fvector_t> xTrain;
	vector<fvector_t> yTrain;
	vector<fvector_t> xTest;
	vector<fvector_t> yTest;

	if (!LoadData(xTrain, yTrain, "train.csv"))
	{
		printf("Failed loading Training data \n\n");
		return -1;
	}

	if (!LoadData(xTest, yTest, "test.csv"))
	{
		printf("Failed loading Test data \n\n");
		return -1;
	}

	printf("Loaded %zu X Training entries \n\n", xTrain.size());
	printf("Loaded %zu Y Training entries \n\n", yTrain.size());
	printf("Loaded %zu X Test entries \n\n", xTest.size());
	printf("Loaded %zu Y Test entries \n\n", yTest.size());

	// perform one hot encoding of train/test labels
	vector<fvector_t> encodedYTrain = yTrain;
	vector<fvector_t> encodedYTest = yTest; 

	// prepare a 3 layer ANN
	shared_ptr<XNeuralNetwork> net = make_shared<XNeuralNetwork>();

	net->AddLayer(make_shared<XFullyConnectedLayer>(6, 4));
	net->AddLayer(make_shared<XReLuActivation>());
	net->AddLayer(make_shared<XFullyConnectedLayer>(4, 4));
	net->AddLayer(make_shared<XReLuActivation>());
	net->AddLayer(make_shared<XFullyConnectedLayer>(4, 3));
	net->AddLayer(make_shared<XSoftMaxActivation>());

	// create training context with Nesterov optimizer and Cross Entropy cost function
	shared_ptr<XNetworkTraining> netTraining = make_shared<XNetworkTraining>(net,
		make_shared<XAdamOptimizer>(0.1f),
		make_shared<XCrossEntropyCost>());

	XClassificationTrainingHelper trainingHelper(netTraining, argc, argv);
	trainingHelper.SetTestSamples(xTest, encodedYTest);
	trainingHelper.SetInputFileName(FILE_NAME);
	trainingHelper.SetOutputFileName(FILE_NAME);
	trainingHelper.SetSaveMode(NetworkSaveMode::OnValidationImprovement);

	// 40 epochs, 10 samples in batch
	trainingHelper.RunTraining(40, 10, xTrain, encodedYTrain);

	return 0;
}

int main(int argc, char** argv)
{
	while (true)
	{
		cout << "Press ENTER to compute values from the current model.\n";
		cout << "Press C to clear the current model.";

		string ln;
		getline(cin, ln);
		transform(ln.begin(), ln.end(), ln.begin(), ::tolower);

		if (ln == "c")
		{
			remove(FILE_NAME.c_str());
		}
		else
		{
			train(argc, argv);
		}
	}
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
