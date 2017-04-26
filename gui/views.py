from django.shortcuts import render
from django.http import JsonResponse
import gui.core.RNN_predict_func as RNN
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def analysis(request):
	if(request.method == 'POST'):
		lines = request.POST['data'].split('\n')
		op=RNN.RNN_predict(lines)
		finalPercentage=RNN.calcFinalPercentage(op)
		data=list(zip(lines, list(map(int, op))))
		#data = {}
		#for i,line in enumerate(lines):
		#	data[line] = int(op[i]) 
		return JsonResponse({'data': data, 'fP': finalPercentage})
	return render(request, "analysis.html")