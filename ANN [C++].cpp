//============================================================================
// Name        : Main.cpp
// Author      : Akbar Telikani
// Copyright   : October 2018
// Description : Object Oriented ProgrammingI by C++
//============================================================================
#include <iostream>
#include<conio.h>
#include<iomanip>
#include <algorithm>
#include <math.h>
#include <fstream>
#include <bits/stdc++.h>
#include <string> 
#include <time.h>
#include <windows.h>
using namespace std;

class ANN
{
	private:
		int NumFeatures;
		vector <vector <double> > InputData;
		vector <vector <double> > TrainData;
		vector <vector <double> > TestData;
		//vectors for saving weight values
		vector <vector <double> > InputWeights;
		double HiddenWeights[20][20][20];	//[units][units][layers]
		vector <double> OutputWeights;
		double UpdateArr[3][500];			// Out.Theta.Error
		vector <double> Out;				//All inputs
		double LRate;
		int Rounds, Units, Layers;
		vector <double> newData;
	public:
		void Menu();
		void Sampling(float TrainRatio);
		void split(string str, string delimiter, vector<double> &v);
		void Normalization(vector <vector <double> > &Data);
		void Initialization(int layers, int units);
		string convertFltStr(double param);
		void BackPropagation(int layers, int units, double rate,int rounds,double Tolerance);
		void Write();
		double Test(int layers, int units, vector <vector <double> > &Data);
		double TestMediator(int layers, int units);
		double activate(double in);
		double activateFirst(double in);
		int Random();
		void Read();
};
//==============================================================
//     Reading data file
//==============================================================
void ANN::Read()
{
	string line;
	string delimiter = ",";
	ifstream InputFile ("Dataset-1.txt");
	if (InputFile.peek() != std::ifstream::traits_type::eof())
	{
		while ( getline (InputFile,line) )
		{
			split(line,delimiter, newData);
			InputData.push_back(newData);
			newData.clear();
		}
	}	
	InputFile.close();
	Normalization(InputData);
}
//==============================================================
//     Writing Test and Train vectors
//==============================================================
void ANN::Write()
{
	ofstream myfile1 ("Train.txt");
	ofstream myfile2 ("Test.txt");
	string str="";
	
	for(int i = 0; i < TrainData.size(); i ++)
	{
		int j=0;
		while(j<=TrainData[i].size()-2)
		{
			str+=convertFltStr(TrainData[i][j])+",";
			j++;
		}
		str+=convertFltStr(TrainData[i][j]);
    	myfile1 << str<<"\n";
    	str="";
	}
	myfile1.close();
	
	for(int i = 0; i < TestData.size(); i ++)
	{
		int j=0;
		while(j<=TestData[i].size()-2)
		{
			str+=convertFltStr(TestData[i][j])+",";
			j++;
		}
		str+=convertFltStr(TestData[i][j]);
    	myfile2 << str<<"\n";
    	str="";
	}
	myfile2.close();	
}
//==============================================================
//     Seperating the Train data from the Test data
//==============================================================
void ANN::Sampling(float TrainRatio)
{
	vector <int> Index;
	TrainData.clear();
	TestData.clear();
	int TrainInt=InputData.size()*TrainRatio;
	int TestInt=InputData.size()-TrainInt;
	int value=0;
	for(int i = 0; i < TrainInt; i++)
	{
		value=rand() % InputData.size()-1;
		value=Random();
		newData=InputData[value];
		Index.push_back(value);
		TrainData.push_back(newData);
		newData.clear();
	}
	for(int i = 0; i < InputData.size(); i++)
	{
		bool find=false;
		for(int j = 0; j < Index.size(); j++)
		{
			if(Index[j]==i)
			{
				find=true;
				break;
			}
		}
		if(find==false)
		{
			newData=InputData[i];
			TestData.push_back(newData);
			newData.clear();
		}
	}
	Write();
}
//==============================================================
//     Generating random value
//==============================================================
int ANN::Random()
{
	int rnd;
	rnd=rand() % InputData.size()-1;
	return rnd;
}
//==============================================================
//     Spliting a string using a delimiter
//==============================================================
void ANN::split(string str, string delimiter, vector<double> &v)
{
	char delimiters[40];
	strcpy(delimiters, delimiter.c_str());
	
	char ins[1000]; 
	strcpy(ins, str.c_str());
	char *tok = strtok(ins, delimiters);
	while (0 != tok)
    {
    	str=tok;
    	v.push_back(strtof(str.c_str(),0));
    	tok = strtok(NULL, delimiters); // move to the next one
    }
}
//==============================================================
//     Normalizing values
//==============================================================
void ANN::Normalization(vector <vector <double> > &Data)
{
	vector<float> temp;
	float max, min;
	for (unsigned int j = 0; j < Data[0].size(); j++)
	{
		for (unsigned int i = 0; i < Data.size()-1; i++)
			temp.push_back(200 * ((Data[i+1][j] - Data[i][j]) / (Data[i+1][j] + Data[i][j])));
			
		max = *max_element(temp.begin(), temp.end());
		min = *min_element(temp.begin(), temp.end());
		
		for (int i = 0; i < temp.size(); i++)
			Data[i][j] =2 * ((temp[i] - min) / (max - min)) - 1;
		temp.clear();
	}
	Data.erase(Data.begin()+Data.size()-1);
	// Writing  the normalized dataset into a file
	ofstream myfile1 ("Normal.txt");
	string str="";
	for(int i = 0; i < Data.size(); i ++)
	{
		int j=0;
		while(j<Data[i].size()-1)
		{
			str+=convertFltStr(Data[i][j])+",";
			j++;
		}
		str+=convertFltStr(Data[i][j]);
    	myfile1 << str<<"\n";
    	str="";
	}
	myfile1.close();
}
//==============================================================
//     Converting Float to string
//==============================================================
string ANN::convertFltStr(double param)
{
	ostringstream str;
	str << param;
	string geek = str.str();
	return geek;
}
//==============================================================
//     Backprobagating the Train data
//==============================================================
void ANN::BackPropagation(int layers, int units, double LearningRate, int rounds,double Tolerance)
{
	//double Tolerance=0.02;
	stringstream out1;
	stringstream out2;
	out1<<rounds;
	out2<<layers;
	string str="Output"+out1.str()+"-"+out2.str()+" Layers"+".txt";
	ofstream myfile (str.c_str());
	myfile<<"========Experimental settings=========\n";
	myfile<<"The number of layers: "<<layers<<"\n";
	myfile<<"The number of units: "<<units<<"\n";
	myfile<<"The learning rate: "<<LearningRate<<"\n";
	myfile<<"The maximum cycle: "<<rounds<<"\n";
	myfile<<"The Tolerance value: "<<Tolerance<<"\n";
	myfile<<"===================================\n";
	double Merror=2;
	int Mcycle;
	vector <vector <double> > InputWeightsM;
		double HiddenWeightsM[20][20][20];	//[units][units][layers]
		vector <double> OutputWeightsM;
		double UpdateArrM[3][500];			// Out.Theta.Error
	double SAE=0;
	double AvgErr=1;
	int cycle=1;
	int InputSize=TrainData[0].size()-1;							//The number of input attributes
	int NumNodes=InputSize+(layers*units)+1;						//The number of nodes in the network
	int NumTheta=(units*layers)+1;									//The number of non-input layers
	double OutputError;
	double ErrorTmp;
	double sum=0;
	while(cycle<=rounds && AvgErr>Tolerance)   						//while terminating condition is not satisfied
	{
		double SumAvg=0;
		int ItraCycle=1;
		for (int i = 0; i < TrainData.size(); i++)		    		//for each training tuple X in D
		{
			//**********************************************************************
			//Output of an input unit is its actual input value
			//**********************************************************************
			Out.clear();
			for(int j = 0; j < TrainData[i].size()-1; j++)			//for each input layer unit
				Out.push_back(TrainData[i][j]);				
			//**********************************************************************
			//compute the net output of unit j with respect to the previous layer, i
			//**********************************************************************
			// For each node in the first hidden layer
			for(int j = 0; j <units; j++)
			{
				for(int w = 0; w < InputSize; w++)	
					sum+=InputWeights[j][w]*Out[w];	
				sum+=UpdateArr[1][j];
				UpdateArr[0][j]=activate(sum);	
				sum=0;
			}
			//for each hidden or output layer unit
			int ss=2;
			while(ss!=layers+1 && layers>1)
			{
				for(int k = 0; k < units; k++)
				{
					for(int w = 0; w < units; w++)
						sum+=HiddenWeights[w][k][ss-2]*UpdateArr[0][((ss-2)*units)+w];
					sum+=UpdateArr[1][(ss-1)*units+k];
					UpdateArr[0][(ss-1)*units+k]=activate(sum);		//Theta-j
					sum=0;
				}
				ss++;
			}
			// Output node
			for(int w = 0; w < units; w++)					
				sum+=OutputWeights[w]*UpdateArr[0][(NumTheta-(units+1))+w];
			sum+=UpdateArr[1][NumTheta-1];
			UpdateArr[0][NumTheta-1]=activate(sum);
			sum=0;
			//**********************************************************************
			//Backpropagate the errors:
			//**********************************************************************
			UpdateArr[2][NumTheta-1]= activateFirst(UpdateArr[0][NumTheta-1])*(TrainData[i][TrainData[i].size()-1]-UpdateArr[0][NumTheta-1]);//compute the output error	
			int LastIndex=units-1;	
			for(int k = NumTheta-2; k>=0; k--)//for each unit j in the hidden layers, from the last to the first hidden layer
			{
				ErrorTmp=activateFirst(UpdateArr[0][k]);
				int CL=k/units;
				if(CL==layers-1)//The last hidden layer
				{
					UpdateArr[2][k]=ErrorTmp*(UpdateArr[2][NumTheta-1]*OutputWeights[LastIndex]);//compute the error with respect to the output layer
					LastIndex--;
				}
				else //The other hidden layers
				{
					int index=(CL+1)*units;
					int row=k-(CL*units);
					for(int w = 0; w < units; w++)
						sum+=UpdateArr[2][index+w]* HiddenWeights[row][w][CL];
					UpdateArr[2][k]=ErrorTmp*sum;
					
					sum=0;
				}
			}
			//**********************************************************************
			//Weight increment and update
			//**********************************************************************
			float delta;
			//for the weights between input layer and the first hidden layer
			for(int w =0; w <InputSize; w++)//equals to the number of input attributes
			{
				for(int k =0; k<units; k++)
				{
					delta=InputWeights[k][w]+(LearningRate*UpdateArr[2][k]*Out[w]);//weight update
					InputWeights[k][w]=delta;
					delta=0;	
				}	
			}
			//for the weights between the last hidden and output layers
			for(int k =0; k<units; k++)
			{
				delta=OutputWeights[k]+(LearningRate*UpdateArr[2][NumTheta-1]*UpdateArr[0][NumTheta-2-k]);//weight update
				OutputWeights[k]=delta;
				delta=0;
			}
			//for the weights between the hidden layers
			int index=0;
			for(int j = 0; j < layers-1; j++)
			{
				index=j*units;
				for(int k =0; k<units; k++)
				{
					for(int w =0; w <units; w++)
					{
						delta=HiddenWeights[k][w][j]+(LearningRate*UpdateArr[2][index+units+w]*UpdateArr[0][index+k]);//weight update
						HiddenWeights[k][w][j]=delta;
						delta=0;
					}	
				}
			}
			//**********************************************************************
			//Bias increment and update
			//**********************************************************************
			for(int k =0; k< NumTheta; k++)
			{
				delta=UpdateArr[1][k]+LearningRate*UpdateArr[2][k];
				UpdateArr[1][k]=delta;
				delta=0;
			}
			//SumAvg+=abs(TrainData[i][TrainData[i].size()-1]- UpdateArr[0][NumTheta-1]);
			ItraCycle++;
		}
		AvgErr=Test(layers, units,TrainData);
		//AvgErr=SumAvg/ItraCycle;
		SAE+=AvgErr;
		cout<<"Cycle "<<cycle<<": "<<AvgErr<<"\n";
		myfile<<"Cycle "<<cycle<<": "<<AvgErr<<"\n";
		if(AvgErr<Merror)
		{
		     Merror=AvgErr;
		     Mcycle=cycle;
		      InputWeightsM=InputWeights;
		     OutputWeightsM=OutputWeights;
		     for(int j = 0; j < units; j++)
			{
				for(int i = 0; i < layers-1; i++)
				{
					for(int k = 0; k < units; k++)
						HiddenWeightsM[j][k][i]=HiddenWeights[j][k][i];
			    }
		    }
    		for(int i = 0; i < NumTheta; i++)
				UpdateArrM[1][i]=UpdateArr[1][i];
	 	}
		cycle++;
	}
	cout<<"Min average error="<<Merror<<" was found in cycle no."<<Mcycle;
	cout<<"\n  Total average of all cycle Error: "<<SAE/cycle;
	myfile<<"Total average Error: "<<SAE/cycle;
	myfile<<"\n Min average Error: "<<Merror;
	myfile.close();
	// Reload Min average
	InputWeights=InputWeightsM;
	OutputWeights=OutputWeightsM;
	for(int j = 0; j < units; j++)
	{
		for(int i = 0; i < layers-1; i++)
		{
			for(int k = 0; k < units; k++) 
				HiddenWeights[j][k][i]=HiddenWeightsM[j][k][i];
	    }
    }
    for(int i = 0; i < NumTheta; i++)//generating Theta
		UpdateArr[1][i]=UpdateArrM[1][i];
	getche();
	//**********************************************************************
	//Writing Bias values to file
	//**********************************************************************	
	ofstream myfile1 ("Bias.txt");
	string Name="";
	for(int i = 0; i < NumTheta/units; i ++)
	{
		stringstream out;
	    out<<i+1;
	    Name="Hidden Layer"+out.str();
		myfile1 <<Name<<"\n";
		for(int j = i*units; j < i*units+units; j ++)
    		myfile1 << convertFltStr(UpdateArr[1][j])<<"\n";
	}
	myfile1 <<"Target Bias"<<"\n";
	myfile1 << convertFltStr(UpdateArr[1][NumTheta-1]);
	myfile1.close();
	//**********************************************************************
	//Writing the weights to file
	//**********************************************************************	
	ofstream myfile2 ("Weights.txt");
	str="";
	for(int i = 0; i < InputSize; i++)
	{
		stringstream out;
	    out<<i+1;
	    str="Input "+out.str()+"  ";
		int j=0;
		while(j<units-1)
		{
			str+=convertFltStr(InputWeights[j][i])+",";
			j++;
		}
		str+=convertFltStr(InputWeights[j][i]);
    	myfile2 << str<<"\n";
    	str="";
	}
	for(int k = 0; k < layers-1; k ++)
	{
		myfile2 <<"======================================"<<"\n";
		for(int i = 0; i < units; i ++)
		{
			stringstream out;
		    out<<i+1;
		    str="N "+out.str()+"  ";
			int j=0;
			while(j<units-1)
			{
				str+=convertFltStr(HiddenWeights[i][j][k])+",";
				j++;
			}
			str+=convertFltStr(HiddenWeights[i][j][k]);
	    	myfile2 << str<<"\n";
	    	str="";
		}		
	}
	myfile2 <<"======================================"<<"\n";
	for(int i = 0; i < units; i ++)
		myfile2 << OutputWeights[i]<<"\n";
	myfile2.close();
}
//==============================================================
//     Testing the data
//==============================================================
double ANN::TestMediator(int layers, int units)
{
return Test(layers, units,TestData);
}

double ANN::Test(int layers, int units,vector <vector <double> > &Data)
{
	int InputSize=TrainData[0].size()-1;							//The number of input attributes
	int NumNodes=InputSize+(layers*units)+1;						//The number of nodes in the network
	int NumTheta=(units*layers)+1;									//The number of non-input layers
	double OutputError;
	double SumError=0;
	double ErrorTmp;
	double sum=0;
	for (int i = 0; i < Data.size(); i++)		    		//for each training tuple X in D
	{
		//**********************************************************************
		//Output of an input unit is its actual input value
		//**********************************************************************
		Out.clear();
		for(int j = 0; j < Data[i].size()-1; j++)			//for each input layer unit
			Out.push_back(Data[i][j]);					
		//**********************************************************************
		//compute the net output of unit j with respect to the previous layer, i
		//**********************************************************************
		// For each node in the first hidden layer
		for(int j = 0; j <units; j++)
		{
			for(int w = 0; w < InputSize; w++)	
				sum+=InputWeights[j][w]*Out[w];		
			sum+=UpdateArr[1][j];
			UpdateArr[0][j]=activate(sum);	
			sum=0;				
		}
		//for each hidden or output layer unit
		int ss=2;
		while(ss!=layers+1 && layers>1)
		{
			for(int k = 0; k < units; k++)
			{
				for(int w = 0; w < units; w++)
					sum+=HiddenWeights[w][k][ss-2]*UpdateArr[0][((ss-2)*units)+w];
				sum+=UpdateArr[1][(ss-1)*units+k];
				UpdateArr[0][(ss-1)*units+k]=activate(sum);		//Theta-j
				sum=0;
			}
			ss++;
		}
		// Output node
		for(int w = 0; w < units; w++)					
			sum+=OutputWeights[w]*UpdateArr[0][(NumTheta-(units+1))+w];
		sum+=UpdateArr[1][NumTheta-1];
		UpdateArr[0][NumTheta-1]=activate(sum);
		sum=0;
		//**********************************************************************
		//Computing the output errors:
		//**********************************************************************
		OutputError= Data[i][Data[i].size()-1]-UpdateArr[0][NumTheta-1];//compute the output error
		if(Data.size()==TestData.size())	
			cout<<"Instance "<<i+1<<": Error value :"<<OutputError<<"\n";
		SumError+=abs(OutputError);
	}
	return SumError/Data.size();
}
double ANN::activate(double in) { return 2 / (1 + exp(-in)) - 1; }
double ANN::activateFirst(double in) { return 0.5*(1 + activate(in))*(1 - activate(in)); }
//double activateFirst(double in) { return 0.5*(1 + activate(in))*(1 - activate(in)); }
//==============================================================
//     Initializing all weights and biases in network
//     layers=hidden layers
//==============================================================
void ANN::Initialization(int layers, int units)
{
	Out.clear();
	memset(HiddenWeights, 0, 8000*(sizeof(double)));
	memset(UpdateArr, 0, 1500*(sizeof(double)));
	InputWeights.clear();
	OutputWeights.clear();
	
	vector<double> temp;
	int InputSize=TrainData[0].size()-1;
	int NumNodes=InputSize+(layers*units)+1;
	int NumTheta=NumNodes-InputSize;
	int rnd;
	for(int j = 0; j < units; j++)
	{
		for(int i = 0; i < InputSize; i++)//Rows are the units and columns are the inputs
		{
			rnd=rand() % 99;
			if(rand() % 10>5)
				temp.push_back(((double) rnd/100)*-1);
			else
				temp.push_back((double) rnd/100);
		}
		InputWeights.push_back(temp);
		
		temp.clear();
		if(layers>1)
		{
			for(int i = 0; i < layers-1; i++)
			{
				for(int k = 0; k < units; k++)
				{
					rnd=rand() % 99;
					if(rand() % 10>5)
						HiddenWeights[j][k][i]=((double) rnd/100)*-1;
					else
						HiddenWeights[j][k][i]=(double) rnd/100;
				}
			}
		}
		
		rnd=rand() % 99;
		if(rand() % 10>5)
			OutputWeights.push_back(((double) rnd/100)*-1);
		else
			OutputWeights.push_back((double) rnd/100);
		
	}
	for(int i = 0; i < NumTheta; i++)//generating Theta
	{
		rnd=rand() % 50;
		if(rand() % 10>5)
			UpdateArr[1][i]=(double) rnd/100;
		else
			UpdateArr[1][i]=((double) rnd/100)*-1;
	}
}
//==============================================================
//     Showing main menu
//==============================================================
void ANN::Menu()
{
		system("CLS");
		cout<<" \n\t\t====================================================";
 		cout << "\n\t\t\t\t ANN:	Main Menu";                                      
 		cout<<" \n\t\t====================================================";
     	cout << "\n\n\t\t  1. Sampling";
    	cout << "\n\n\t\t  2. Training";
    	cout << "\n\n\t\t  3. Testing";
     	cout << "\n\n\t\t  4. Exit Program";
     	cout<<  "\n\t\t===================================================";
     	cout << "\n\n";
     	cout << "\t\t Select Your Choice:";
}
//==============================================================
//    Main Program
//==============================================================
int main(int argc, char** argv) 
{
	ANN Net;
	double LRate,Tolerance;
	int Rounds, Units, Layers;
	float TrainRatio=0;
	Net.Read();
	MainPoint:
	int choice;
	while(1) 
	{
		Net.Menu();
		
		cin>>choice;
     	switch(choice)
     	{
     		case 1://Random Sampling
     		{
				cout << "\n\n\t\t  Enter Train Ratio [e.g 0.8]:";
				cin>>TrainRatio;
				
				ostringstream str1;
				str1 << TrainRatio;
				string geek = str1.str();
				string ss="back";
				if (strcasecmp(geek.c_str(), ss.c_str()) == 0)
					goto MainPoint;
				else
					Net.Sampling(TrainRatio);
				break;
     		}
 			case 2://Training
 			{
 				if(TrainRatio!=0)
 				{
 					cout << "\n\t\t  Enter Rounds:";
					cin>>Rounds;
		       		cout << "\n\t\t  Enter Learning Rate [e.g 0.01]:";
		       		cin>>LRate;
		       		cout << "\n\t\t  Enter Tolerance error[e.g 0.03]:";
		       		cin>>Tolerance;
		       		cout << "\n\t\t  Enter the Number of Hidden layers:";
		       		cin>>Layers;
					cout << "\n\t\t  Enter the Number of Units:";
		       		cin>>Units;
		       		Net.Initialization(Layers, Units);
		       		Net.BackPropagation(Layers, Units, LRate, Rounds,Tolerance);
 				}
 				else
 					cout << "\n\t\t  Please perform the sampling phase!!!";
 				break;
 			}
 			case 3://Testing
 			{
 				if(Rounds==0)
 					cout << "\n\t\t  Please perform the training phase!!!";
 				else
 					cout<<"\n\t\t Average Error: "<<Net.TestMediator(Layers, Units);
 				break;
 			}
     		case 4:
			{
			   	system("CLS");
			   	exit(0);
	   			break;
	   		}
	   		default : 
			{
				cout<<"\n\n\t\t Invalid input";
				cout<<"\n\n\t\tPress Enter to continue";
	   		}
     	}
     	getche();
	}
	return 0;
}
