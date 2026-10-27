import itertools
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

st.set_page_config(page_title='Pairwise Exchange Layout', layout='wide')

def objective(layout, flow, distance):
    # layout[position] is the machine occupying that position.
    pos = np.empty(len(layout), dtype=int)
    for p, machine in enumerate(layout):
        pos[machine] = p
    return float(np.sum(flow * distance[np.ix_(pos, pos)]))

def solve(flow, distance, initial):
    current = tuple(initial)
    history = [{'iteration': 0, 'layout': current, 'objective': objective(current, flow, distance), 'exchange': None}]
    evaluations = []
    while True:
        old = history[-1]['objective']
        candidates = []
        # Pair labels refer to machine numbers, not position numbers.
        for a, b in itertools.combinations(range(len(current)), 2):
            trial = list(current)
            i, j = trial.index(a), trial.index(b)
            trial[i], trial[j] = trial[j], trial[i]
            val = objective(trial, flow, distance)
            candidates.append({'Machine A': a+1, 'Machine B': b+1, 'Candidate layout': ' – '.join(str(x+1) for x in trial), 'Objective': val, 'Change': val-old, 'Reduction': old-val, '_layout': tuple(trial)})
        best = min(candidates, key=lambda x: x['Objective'])
        improved = best['Objective'] < old - 1e-9 * max(1, abs(old))
        for row in candidates:
            row['Selected'] = bool(improved and row is best)
        evaluations.append({'iteration': history[-1]['iteration'], 'layout': current, 'objective': old, 'candidates': candidates})
        if not improved:
            break
        current = best['_layout']
        history.append({'iteration': len(history), 'layout': current, 'objective': best['Objective'], 'exchange': (best['Machine A'], best['Machine B'])})
    return history, evaluations

def plot_layout(layout):
    n = len(layout)
    fig, ax = plt.subplots(figsize=(max(7, min(n * 1.35, 20)), 2.3))
    for i, m in enumerate(layout):
        ax.add_patch(Rectangle((i, 0), 1, 1, facecolor='#dbeafe', edgecolor='#1e3a5f', linewidth=2))
        ax.text(i+.5, .55, f'Machine {m+1}', ha='center', va='center', fontsize=11, weight='bold')
        ax.text(i+.5, -.18, f'Position {i+1}', ha='center', va='center', fontsize=9)
    ax.set_xlim(0, n); ax.set_ylim(-.4, 1.2); ax.set_aspect('equal'); ax.axis('off')
    fig.tight_layout()
    return fig

def label_layout(layout):
    return ' – '.join(str(x+1) for x in layout)

def matrix_input(df, key, label):
    st.markdown(f'**{label}**')
    return st.data_editor(df, key=key, num_rows='fixed', use_container_width=True,
        column_config={col: st.column_config.NumberColumn(col, min_value=0.0, format='%.3f', required=True) for col in df.columns})

st.title('Pairwise Exchange Layout Optimizer')
st.write('Improve a layout by testing all two-machine exchanges at each iteration.')
with st.expander('How the matrices work', expanded=False):
    st.markdown('''**Flow:** Row = origin machine; column = destination machine. Enter each actual direction separately. A single upper-triangular value is counted once; if both directions have values, both are counted. Equal values in opposite cells are treated as two separate flows, not as duplicate entries.

**Distance:** Row = origin position; column = destination position. Distances belong to fixed positions, not machines. You may enter symmetric or directional distances. The objective is **Σ flow[i, j] × distance[position of i, position of j]**. For a single combined flow per pair with directional distances, the direction of the cell used matters.

The layout graphic shows position order only. Its boxes do not represent the geometry or scale of a custom distance matrix.''')

n = st.number_input('Number of machines / positions', min_value=2, max_value=30, value=4, step=1)
n = int(n)
names = [f'M{i+1}' for i in range(n)]
positions = [f'P{i+1}' for i in range(n)]
example = st.checkbox('Load the four-machine example from the slides', value=(n == 4), disabled=(n != 4), key=f'example_{n}')
base_f = np.zeros((n,n))
if n == 4 and example:
    base_f[0,1],base_f[0,2],base_f[0,3],base_f[1,2],base_f[1,3],base_f[2,3] = 10,15,20,10,5,5
base_d = np.abs(np.arange(n)[:,None] - np.arange(n)[None,:]).astype(float)
flow_df = pd.DataFrame(base_f, index=names, columns=names)
dist_df = pd.DataFrame(base_d, index=positions, columns=positions)
st.subheader('1. From–to flow chart')
st.caption('Rows and columns are machines. Use zero for no flow. Enter both directions only when both flows exist.')
f = matrix_input(flow_df, f'flow_{n}_{example}', 'Flow matrix')
st.subheader('2. Distance between positions')
st.caption('The default is a one-unit-spaced linear layout. Edit any position-to-position distance as needed.')
d = matrix_input(dist_df, f'dist_{n}_{example}', 'Position distance matrix')
st.subheader('3. Initial layout')
st.caption('Enter every machine number once, from position 1 to the last position.')
initial_text = st.text_input('Initial machine order (comma separated)', ', '.join(map(str, range(1,n+1))), key=f'initial_{n}')
if st.button('Run pairwise exchange', type='primary'):
    try:
        flow = f.to_numpy(dtype=float); distance = d.to_numpy(dtype=float)
        if not np.isfinite(flow).all() or not np.isfinite(distance).all() or (flow<0).any() or (distance<0).any():
            raise ValueError('All matrix values must be finite, nonnegative numbers.')
        if not np.allclose(np.diag(flow), 0) or not np.allclose(np.diag(distance), 0):
            raise ValueError('The diagonals of both matrices must be zero.')
        initial = [int(x.strip())-1 for x in initial_text.split(',')]
        if sorted(initial) != list(range(n)):
            raise ValueError('Initial layout must contain every machine number exactly once.')
        history, evaluations = solve(flow, distance, initial)
        st.session_state['result'] = {'history': history, 'evaluations': evaluations, 'flow': flow, 'distance': distance}
    except (ValueError, TypeError) as e:
        st.error(f'Please check the inputs: {e}')

if 'result' in st.session_state:
    r = st.session_state['result']; history = r['history']; evaluations = r['evaluations']; final = history[-1]
    st.divider(); st.header('Results')
    st.caption('Results reflect the inputs at the most recent run. Run the algorithm again after changing an input.')
    c1,c2,c3 = st.columns(3)
    c1.metric('Final objective value', f"{final['objective']:,.4f}")
    c2.metric('Initial objective value', f"{history[0]['objective']:,.4f}")
    c3.metric('Improvement', f"{history[0]['objective']-final['objective']:,.4f}")
    st.write('**Final layout:** ' + label_layout(final['layout']))
    fig = plot_layout(final['layout']); st.pyplot(fig); plt.close(fig)
    st.info('Stopped when no single exchange reduced the objective. This is a pairwise local optimum; a global optimum is not guaranteed.')
    with st.expander('Objective values for all iterations'):
        table = pd.DataFrame([{'Iteration': h['iteration'], 'Layout': label_layout(h['layout']), 'Exchange accepted': 'Initial' if h['exchange'] is None else f"{h['exchange'][0]} ↔ {h['exchange'][1]}", 'Objective': h['objective']} for h in history])
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.line_chart(table.set_index('Iteration')['Objective'])
        st.download_button('Download iteration history (CSV)', table.to_csv(index=False).encode(), 'iterations.csv', 'text/csv')
    with st.expander('Exchanges considered in an iteration'):
        idx = st.selectbox('Choose current iteration to inspect', range(len(evaluations)), format_func=lambda i: f"Iteration {evaluations[i]['iteration']} — layout {label_layout(evaluations[i]['layout'])} — objective {evaluations[i]['objective']:g}")
        ev = evaluations[idx]
        rows = [{k:v for k,v in row.items() if k != '_layout'} for row in ev['candidates']]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption('Change = candidate objective − current objective; reduction = current objective − candidate objective. A positive reduction improves the layout. The last iteration shows the stopping check.')
    with st.expander('Final objective calculation'):
        flow, distance = r['flow'], r['distance']; layout = final['layout']; pos = np.empty(len(layout), dtype=int)
        for p,m in enumerate(layout): pos[m]=p
        details=[]
        for i in range(len(layout)):
            for j in range(len(layout)):
                if i != j and flow[i,j] != 0:
                    details.append({'From': f'M{i+1}', 'To': f'M{j+1}', 'Flow': flow[i,j], 'From position': f'P{pos[i]+1}', 'To position': f'P{pos[j]+1}', 'Distance': distance[pos[i],pos[j]], 'Flow × distance': flow[i,j]*distance[pos[i],pos[j]]})
        if details: st.dataframe(pd.DataFrame(details), hide_index=True, use_container_width=True)
        else: st.write('All flows are zero; the objective is zero.')
