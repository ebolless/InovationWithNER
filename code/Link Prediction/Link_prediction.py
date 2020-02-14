import codecs
import re
import itertools
import networkx as nx
import matplotlib.pyplot as plt

def coacore(sentences):
    word_total = []
    words = set()
    mates = {}
    mates2 = {}
    jacard ={}
    jacard2 ={}
    train_types ={'O':0,'B-APPLICATION':0,'B-ALGORITHM':0,'I-APPLICATION':0,'I-ALGORITHM':0,'sentences': 0}
    for sent in sentences:
        apps = []
        app = ''
        algs = []
        alg = ''
        train_types['sentences'] = train_types['sentences']+1
        for word in sent:
            if word[1] == 'B-APPLICATION':
                if app !='':
                    apps.append(app)
                app = word[0].lower()
            elif word[1] == 'I-APPLICATION':
                app = app + ' '+word[0].lower()
            elif word[1] == 'B-ALGORITHM':
                if alg !='':
                    algs.append(alg)
                alg = word[0].lower()
            elif word[1] == 'I-ALGORITHM':
                alg = alg + ' '+word[0].lower()
            train_types[word[1]] = train_types[word[1]]+1
            word_total.append(word[0])
        if app !='':
            apps.append(app)
        if alg !='':
            algs.append(alg)
        #mate = list(map(';'.join, itertools.chain(itertools.product(apps, algs))))
        mate = list(itertools.chain(itertools.product(apps, algs)))
        if len(mate)>0:
            for mat in mate:
                if mat in mates.keys():
                    mates[mat] = mates[mat] + 1
                else:
                    mates[mat] = 1
                    if mat[0] not in jacard.keys():
                        jacard[mat[0]] = []
                    l = jacard[mat[0]]
                    l.append(mat[1])
                    jacard[mat[0]] = l
                    #print(mat[0],mat[1])
                    #print(jacard[mat[0]])
        mate2 = list(itertools.chain(itertools.product(algs,apps)))
        if len(mate2)>0:
            for mat2 in mate2:
                if mat2 in mates2.keys():
                    mates2[mat2] = mates2[mat2] + 1
                else:
                    mates2[mat2] = 1
                    if mat2[0] not in jacard2.keys():
                        jacard2[mat2[0]] = []
                    l = jacard2[mat2[0]]
                    l.append(mat2[1])
                    jacard2[mat2[0]] = l
                    #print(mat[0],mat[1])
                    #print(jacard[mat[0]])

    words = set(word_total)
    return mates,words,word_total,jacard,jacard2


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def load_sentences(path, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.

    zeros - Replace digits with 0

    """
    sentences = []
    sentence = []
    counter = 0
    for line in codecs.open(path, 'r', 'utf8'):
        counter += 1
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            # if len(word) < 2:
            #    print(word)
            if len(word) >= 2:
                sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme='iob'):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            # raise Exception('Sentences should be given in IOB format! ' + 'Please check sentence %i:\n%s' % (i, s_str))
            print('Removing Problematic sentence: %i:\n%s' % (i, s_str))
            continue
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """IOB -> IOBES"""
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def mainlist(train_jacard):
    mainlist = list(train_jacard)
    jacard_list = []
    for i in range(len(mainlist)):
        for j in range(len(mainlist)):
            if mainlist[j] == mainlist[i]:
                continue;
            else:
                jacc_ = jaccard_similarity(train_jacard[mainlist[i]], train_jacard[mainlist[j]])
                # print (mainlist[i],mainlist[j],jacc_)
                if jacc_ > 0:
                    jacard_list.append([mainlist[i], mainlist[j], jacc_])
    return mainlist, jacard_list

def add_edge_to_graph(G,jacard):
    s=0
    for key, value in jacard.items():
        s = s + len(value)
        for alg in value:
            G.add_edge(key, alg)
    print(s)
    return G

def add_pred_edge_to_graph(G, G_pred, train_jacard):
    for edge in G.edges(data=True):
        # print(edge)
        # print(edge[2]['weight'])
        for alg in set(train_jacard[edge[1]]) - set(train_jacard[edge[0]]):
            if G_pred.has_edge(edge[0], alg):
                # we added this one before, just increase the weight by one
                G_pred[edge[0]][alg]['weight'] += edge[2]['weight']
            else:
                # new edge. add with weight=1
                G_pred.add_edge(edge[0], alg, weight=edge[2]['weight'])
        for alg in set(train_jacard[edge[0]]) - set(train_jacard[edge[1]]):
            if G_pred.has_edge(edge[1], alg):
                # we added this one before, just increase the weight by one
                G_pred[edge[1]][alg]['weight'] += edge[2]['weight']
            else:
                # new edge. add with weight=1
                G_pred.add_edge(edge[1], alg, weight=edge[2]['weight'])
    return  G_pred


def _at_k(G_pred, mainlist):
    p_at_k = {}
    one_at_k = {}
    for i in range(0, 11):
        print(i)
        top_k_predict = {}
        for node in mainlist:
            edge_candidate = []
            for edge in sorted(G_pred.edges(node, data=True), key=lambda t: t[2].get('weight', 1), reverse=True)[0:i]:
                # print(edge)
                edge_candidate.append(edge[1])
            top_k_predict[node] = edge_candidate
        for node in mainlist:
            if node in test_jacard.keys():
                if node in top_k_predict.keys():
                    if len(set(test_jacard[node]) - set(train_jacard[node])) > 0:
                        if i == 0:
                            p = [0.0] * 11
                            one = [0.0] * 11
                            p[0] = len(set(test_jacard[node]) - set(train_jacard[node]))
                            one[0] = len(set(test_jacard[node]) - set(train_jacard[node]))
                            one_at_k[node] = one
                            p_at_k[node] = p
                            # print(p_at_k[node])
                        else:
                            p = p_at_k[node]
                            one = one_at_k[node]
                            if len(list(set(test_jacard[node]) & set(top_k_predict[node]))) > 0:
                                p[i] = len(list(set(test_jacard[node]) & set(top_k_predict[node]))) / min(p[0], i)
                                one[i] = 1
                                # print(p)

                            else:
                                # print(p[i])
                                p[i] = 0.0
                                one[i] = 0
                            p_at_k[node] = p
                            one_at_k[node] = one
    return p_at_k, one_at_k



def save_graph(avg_one_at_k, pic_name ='OneatN.png', G):
    n = range(1, 11)

    plt.plot(n, avg_one_at_k, color='b')

    plt.xlabel('n')
    plt.ylabel('Precision')
    # plt.title('Average precision @ n')

    plt.savefig(pic_name, format='png', dpi=300, bbox_inches='tight')
    plt.show()

    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]

    pos = nx.spring_layout(G)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=G.number_of_nodes())

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge,
                           width=6)
    nx.draw_networkx_edges(G, pos, edgelist=esmall,
                           width=6, alpha=0.5, edge_color='b', style='dashed')

    # labels
    nx.draw_networkx_labels(G, pos, font_size=5, font_family='sans-serif')

    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    train_data = '/Data/train_IBO_t.txt'
    test_data = '/Data/test_IBO_t.txt'
    dev_data = '/Data/dev_IBO_t.txt'
    zeros = True
    train_sentences = load_sentences(train_data, zeros)
    test_sentences = load_sentences(test_data, zeros)
    update_tag_scheme(train_sentences)
    update_tag_scheme(test_sentences)
    test_mates, test_words, test_word_total, test_jacard, test_jacard2 = coacore(test_sentences)
    train_mates, train_words, train_word_total, train_jacard, train_jacard2 = coacore(train_sentences)

    mainlist, jacard_list = mainlist(train_jacard)
    mainlist2, jacard_list2 = mainlist(train_jacard2)


    G = nx.Graph()
    G_ui = nx.Graph()
    G_ui_test = nx.Graph()
    G_ui_test.add_nodes_from(mainlist)
    G_ui_test.add_nodes_from(mainlist2)
    s = 0
    G_ui_test = add_edge_to_graph(G_ui_test, test_jacard)
    G.add_nodes_from(mainlist)
    G_ui.add_nodes_from(mainlist)
    G_ui.add_nodes_from(mainlist2)
    G.add_weighted_edges_from(jacard_list)
    G_ui = add_edge_to_graph(G_ui, train_jacard)

    preds = nx.jaccard_coefficient(G_ui)

    G_pred = nx.Graph()
    G_pred = add_pred_edge_to_graph(G, G_pred, train_jacard)

    p_at_k, one_at_k = _at_k(G_pred, mainlist)

    denomeritor = len(p_at_k)
    agg_one_at_k = [0.0] * 10
    for key, item in one_at_k.items():
        agg_one_at_k = [x + y for x, y in zip(item[1:11], agg_one_at_k)]
    agg_p_at_k = [0.0] * 10
    for key, item in p_at_k.items():
        agg_p_at_k = [x + y for x, y in zip(item[1:11], agg_p_at_k)]

    avg_p_at_k = [i / denomeritor for i in agg_p_at_k]

    avg_one_at_k = [i / denomeritor for i in agg_one_at_k]

    save_graph(avg_p_at_k, 'PatN.png', G)
    save_graph(avg_one_at_k, 'OneatN.png', G)



