from django.shortcuts import render
from . import clusteringAlgorithms
from django.http import HttpResponse

# Create your views here.
def Kmeans(request):
    try:
        kmeansResult, cluster_plot, elbow_graph = clusteringAlgorithms.Kmeans()
        context = {'result' : kmeansResult, 'cluster_plot' : cluster_plot, 'elbow_graph' : elbow_graph}
        return render(request, 'kMeans.html', context)
    except Exception as e:
        return HttpResponse(e)