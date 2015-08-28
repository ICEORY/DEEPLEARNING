#include <Python.h>
#include <iostream>
using namespace std;
int main()
{
	Py_Initialize();

	PyObject * pModule = NULL;

	PyObject * pFunc = NULL;

	pModule = PyImport_ImportModule("publish");

	pFunc = PyObject_GetAttrString(pModule,"run");

	PyEval_CallObject(pFunc, NULL);
	
	cout<<"done"<<endl;

	Py_Finalize(); 
}