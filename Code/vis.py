import torch
import plotly.offline as plt
import plotly.graph_objs as go

import torch
import plotly.offline as plt
import plotly.graph_objs as go
import numpy as np


name2color = {name_nodes[i]:"#"+''.join([np.random.choice(list('0123456789ABCDEF')) for j in range(6)])
             for i in range(400)}
    
def generate_poincare(model,data, root_exp, root_data, show_feat=False, epoch=0, loss=0):
  
    with open(f'{root_data}/data_hierarchy.tsv','r') as f:
        edgelist = [line.strip().split('\t') for line in f.readlines()]
        
    with open(f'{root_data}/data_feat.tsv','r') as f:
        featlist = [line.strip().split('\t') for line in f.readlines()]
        
    vis = model.embedding.weight.cpu().data.numpy()
  
  
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5,color='#888'),
        hoverinfo='none',
        mode='lines')
    
    feat_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5,color='#d43928'),
        hoverinfo='none',
        mode='lines')
  
    xs = []
    ys = []
    for s0,s1 in edgelist:
        x0, y0 = vis[data.item2id[s0]]
        x1, y1 = vis[data.item2id[s1]]
  
        xs.extend(tuple([x0, x1, None]))
        ys.extend(tuple([y0, y1, None]))
        
    xs_feat = []
    ys_feat = []
    for s0,s1 in featlist:
        x0, y0 = vis[data.item2id[s0]]
        x1, y1 = vis[data.item2id[s1]]
  
        xs_feat.extend(tuple([x0, x1, None]))
        ys_feat.extend(tuple([y0, y1, None]))
  
    edge_trace['x'] = xs
    edge_trace['y'] = ys
    
    feat_trace['x'] = xs_feat
    feat_trace['y'] = ys_feat
  

  
    xs = []
    ys = []
    names = []
    node_traces = []
    depths = np.array([int(name.split('_')[-2]) for name in data.items])
    name_nodes = list(set([name.split('_')[0] for name in data.items]))
    sizes = 1.5**(np.max(depths) - depths) + 2
    max_size = 100
    sizes = sizes / np.max(sizes) * max_size
    
    number_of_colors = len(name_nodes)

    
    for i,name in enumerate(data.items):
        x, y = vis[data.item2id[name]]
        xs.extend(tuple([x]))
        ys.extend(tuple([y]))
        names.extend(tuple([name.split('_')[0]]))
        threshold = name.split('_')[-1]
        name_node = name.split('_')[0]
        
        info_node = f'{name_node}, depth: {depths[i]}, threshold: {threshold}'
        size = sizes[i]
        
        node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            reversescale=True,
            color=name2color[name_node],
            size=size),legendgroup=i)
  
        node_trace['x'] = [x]
        node_trace['y'] = [y]
        
        node_trace['text'] = info_node
        
        node_traces.append(node_trace)
    node_traces = node_traces[::-1]
        
  
    display_list = np.random.choice(data.items, 1)
    display_list = data.items
    display_list = []
  
    label_trace = go.Scatter(
        x=[],
        y=[],
        mode='text',
        text=[],
        textposition='top center',
        textfont=dict(
            family='sans serif',
            size=13,
            color = "#000000"
        )
    )
  
    for name in display_list:
        x,y = vis[data.item2id[name]]
        label_trace['x'] += tuple([x])
        label_trace['y'] += tuple([y])
        label_trace['text'] += tuple([name.split('_')[0]])
  
  
    data_trace = [edge_trace, *node_traces,label_trace] 
    if show_feat:
        data_trace.append(feat_trace)
    fig = go.Figure(data=data_trace,
                 layout=go.Layout(
                    title=f'Poincare Embedding of Decision Tree, Loss: {loss:4f}',
                    width=700,
                    height=700,
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    fig.add_shape(type="circle",
        xref="x", yref="y",
        x0=-1, y0=-1, x1=1, y1=1,
        line_color="LightSeaGreen",
    )
    dst = f'{root_exp}/vis/poincare embedding_{epoch:5d}.png'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    fig.write_image(dst)
    
def generate_report(model,data, root_exp, root_data):
    print('Generating report...')
  
    x,y = zip(*enumerate(model.log))
  
    trace = go.Scatter(x = x,
                       y = y
                      )
    layout = go.Layout(
                 yaxis=dict(
                      title= 'Loss',
                      ticklen= 5,
                      gridwidth= 2,
                      ),
                 xaxis=dict(
                      title= 'Epoch',
                      ticklen= 5,
                      gridwidth= 2,
                     ))
  
    fig = go.Figure([trace],layout=layout)
    plt.plot(fig,filename=f'{root_exp}/log_loss.html')
  
  
  
    with open(f'{root_data}/data_hierarchy.tsv','r') as f:
        edgelist = [line.strip().split('\t') for line in f.readlines()]
        
    with open(f'{root_data}/data_feat.tsv','r') as f:
        featlist = [line.strip().split('\t') for line in f.readlines()]
        
    vis = model.embedding.weight.data.numpy()
  
  
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5,color='#888'),
        hoverinfo='none',
        mode='lines')
    
    feat_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5,color='#d43928'),
        hoverinfo='none',
        mode='lines')
  
    xs = []
    ys = []
    for s0,s1 in edgelist:
        x0, y0 = vis[data.item2id[s0]]
        x1, y1 = vis[data.item2id[s1]]
  
        xs.extend(tuple([x0, x1, None]))
        ys.extend(tuple([y0, y1, None]))
        
    xs_feat = []
    ys_feat = []
    for s0,s1 in featlist:
        x0, y0 = vis[data.item2id[s0]]
        x1, y1 = vis[data.item2id[s1]]
  
        xs_feat.extend(tuple([x0, x1, None]))
        ys_feat.extend(tuple([y0, y1, None]))
  
    edge_trace['x'] = xs
    edge_trace['y'] = ys
    
    feat_trace['x'] = xs_feat
    feat_trace['y'] = ys_feat
  
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            reversescale=True,
            color='#8b9dc3',
            size=7)
        )
  
    xs = []
    ys = []
    names = []
    for name in data.items:
        x, y = vis[data.item2id[name]]
        xs.extend(tuple([x]))
        ys.extend(tuple([y]))
        names.extend(tuple([name.split('.')[0]]))
  
    node_trace['x'] = xs 
    node_trace['y'] = ys
        
    node_trace['text'] = names 
  
    display_list = np.random.choice(data.items, 1)
  
    label_trace = go.Scatter(
        x=[],
        y=[],
        mode='text',
        text=[],
        textposition='top center',
        textfont=dict(
            family='sans serif',
            size=13,
            color = "#000000"
        )
    )
  
    for name in display_list:
        x,y = vis[data.item2id[name]]
        label_trace['x'] += tuple([x])
        label_trace['y'] += tuple([y])
        label_trace['text'] += tuple([name.split('.')[0]])
  
  
  
    fig = go.Figure(data=[edge_trace, node_trace,label_trace, feat_trace],
                 layout=go.Layout(
                    title='Poincare Embedding of mammals subset of WordNet',
                    width=500,
                    height=500,
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
  
    plt.plot(fig, filename=f'{root_exp}/poincare embedding.html')
  
    print('report is saves as .html files in demo folder.')

def generate_report_old(model,data, root_exp):
    print('Generating report...')
  
    x,y = zip(*enumerate(model.log))
  
    trace = go.Scatter(x = x,
                       y = y
                      )
    layout = go.Layout(
                 yaxis=dict(
                      title= 'Loss',
                      ticklen= 5,
                      gridwidth= 2,
                      ),
                 xaxis=dict(
                      title= 'Epoch',
                      ticklen= 5,
                      gridwidth= 2,
                     ))
  
    fig = go.Figure([trace],layout=layout)
    plt.plot(fig,filename='demo/log_loss.html')
  
  
  
    with open('data/wordnet/mammal_hierarchy.tsv','r') as f:
        edgelist = [line.strip().split('\t') for line in f.readlines()]
        
    vis = model.embedding.weight.data.numpy()
  
  
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5,color='#888'),
        hoverinfo='none',
        mode='lines')
  
    xs = []
    ys = []
    for s0,s1 in edgelist:
        x0, y0 = vis[data.item2id[s0]]
        x1, y1 = vis[data.item2id[s1]]
  
        xs.extend(tuple([x0, x1, None]))
        ys.extend(tuple([y0, y1, None]))
  
    edge_trace['x'] = xs
    edge_trace['y'] = ys
  
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            reversescale=True,
            color='#8b9dc3',
            size=2)
        )
  
    xs = []
    ys = []
    names = []
    for name in data.items:
        x, y = vis[data.item2id[name]]
        xs.extend(tuple([x]))
        ys.extend(tuple([y]))
        names.extend(tuple([name.split('.')[0]]))
  
    node_trace['x'] = xs 
    node_trace['y'] = ys
        
    node_trace['text'] = names 
  
    display_list = ['placental.n.01',
     'primate.n.02',
     'mammal.n.01',
     'carnivore.n.01',
     'canine.n.02',
     'dog.n.01',
     'pug.n.01',
     'homo_erectus.n.01',
     'homo_sapiens.n.01',
     'terrier.n.01',
     'rodent.n.01',
     'ungulate.n.01',
     'odd-toed_ungulate.n.01',
     'even-toed_ungulate.n.01',
     'monkey.n.01',
     'cow.n.01',
     'welsh_pony.n.01',
     'feline.n.01',
     'cheetah.n.01',
     'mouse.n.01']
  
    label_trace = go.Scatter(
        x=[],
        y=[],
        mode='text',
        text=[],
        textposition='top center',
        textfont=dict(
            family='sans serif',
            size=13,
            color = "#000000"
        )
    )
  
    for name in display_list:
        x,y = vis[data.item2id[name]]
        label_trace['x'] += tuple([x])
        label_trace['y'] += tuple([y])
        label_trace['text'] += tuple([name.split('.')[0]])
  
  
  
    fig = go.Figure(data=[edge_trace, node_trace,label_trace],
                 layout=go.Layout(
                    title='Poincare Embedding of mammals subset of WordNet',
                    width=700,
                    height=700,
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
  
    plt.plot(fig, filename='demo/poincare embedding.html')
  
    print('report is saves as .html files in demo folder.')

if __name__ == '__main__':
    import torch

    model = torch.load('demo/model.pt')
    data = torch.load('demo/data.pt')

    generate_report(model,data)