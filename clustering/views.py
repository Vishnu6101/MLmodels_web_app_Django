from django.shortcuts import render
from . import clusteringAlgorithms

# Create your views here.
def Kmeans(request):
    kmeansResult, cluster_plot, elbow_graph = clusteringAlgorithms.Kmeans()
    context = {'result' : kmeansResult, 'cluster_plot' : cluster_plot, 'elbow_graph' : elbow_graph}
    return render(request, 'kMeans.html', context)