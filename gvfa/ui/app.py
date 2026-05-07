"""
GVFA Experiment Dashboard

A Streamlit-based UI for running GVFA experiments interactively.

Run:
    streamlit run gvfa/ui/app.py
"""

import streamlit as st
import pandas as pd
import altair as alt
from itertools import product
from datetime import datetime

st.set_page_config(
    page_title="GVFA Experiment Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="centered"
)


def render_filters(df: pd.DataFrame, batch_type: str, key_prefix: str = "") -> pd.DataFrame:
    """Render filter widgets and return filtered dataframe."""
    filtered_df = df.copy()

    if batch_type == "Node Classification":
        cols = st.columns(5)
        with cols[0]:
            datasets = ["All"] + sorted(df["dataset"].unique().tolist())
            dataset_filter = st.selectbox("Dataset", datasets, key=f"{key_prefix}_dataset")
        with cols[1]:
            d_values = ["All"] + sorted(df["D"].unique().tolist())
            d_filter = st.selectbox("D", d_values, key=f"{key_prefix}_D")
        with cols[2]:
            layers_values = ["All"] + sorted(df["layers"].unique().tolist())
            layers_filter = st.selectbox("layers", layers_values, key=f"{key_prefix}_layers")
        with cols[3]:
            phi_values = ["All"] + sorted(df["phi"].unique().tolist())
            phi_filter = st.selectbox("phi", phi_values, key=f"{key_prefix}_phi")
        with cols[4]:
            norm_values = ["All"] + sorted(df["normalize"].unique().tolist())
            norm_filter = st.selectbox("normalize", norm_values, key=f"{key_prefix}_norm")

        if dataset_filter != "All":
            filtered_df = filtered_df[filtered_df["dataset"] == dataset_filter]
        if d_filter != "All":
            filtered_df = filtered_df[filtered_df["D"] == d_filter]
        if layers_filter != "All":
            filtered_df = filtered_df[filtered_df["layers"] == layers_filter]
        if phi_filter != "All":
            filtered_df = filtered_df[filtered_df["phi"] == phi_filter]
        if norm_filter != "All":
            filtered_df = filtered_df[filtered_df["normalize"] == norm_filter]
    else:
        cols = st.columns(5)
        with cols[0]:
            d_values = ["All"] + sorted(df["D"].unique().tolist())
            d_filter = st.selectbox("D", d_values, key=f"{key_prefix}_D")
        with cols[1]:
            layers_values = ["All"] + sorted(df["layers"].unique().tolist())
            layers_filter = st.selectbox("layers", layers_values, key=f"{key_prefix}_layers")
        with cols[2]:
            phi_values = ["All"] + sorted(df["phi"].unique().tolist())
            phi_filter = st.selectbox("phi", phi_values, key=f"{key_prefix}_phi")
        with cols[3]:
            pooling_values = ["All"] + sorted(df["pooling"].unique().tolist())
            pooling_filter = st.selectbox("pooling", pooling_values, key=f"{key_prefix}_pooling")
        with cols[4]:
            norm_values = ["All"] + sorted(df["normalize"].unique().tolist())
            norm_filter = st.selectbox("normalize", norm_values, key=f"{key_prefix}_norm")

        if d_filter != "All":
            filtered_df = filtered_df[filtered_df["D"] == d_filter]
        if layers_filter != "All":
            filtered_df = filtered_df[filtered_df["layers"] == layers_filter]
        if phi_filter != "All":
            filtered_df = filtered_df[filtered_df["phi"] == phi_filter]
        if pooling_filter != "All":
            filtered_df = filtered_df[filtered_df["pooling"] == pooling_filter]
        if norm_filter != "All":
            filtered_df = filtered_df[filtered_df["normalize"] == norm_filter]

    return filtered_df


def render_table(df: pd.DataFrame, batch_type: str) -> None:
    """Render results table with best highlighting."""
    if df.empty:
        st.info("No results match the current filters.")
        return

    if batch_type == "Node Classification":
        df_sorted = df.sort_values("accuracy", ascending=False).copy()

        best_flags = []
        for _, row in df_sorted.iterrows():
            dataset_df = df_sorted[df_sorted["dataset"] == row["dataset"]]
            is_best = row["accuracy"] == dataset_df["accuracy"].max()
            best_flags.append("*" if is_best else "")
        df_sorted["best"] = best_flags

        display_cols = ["dataset", "D", "layers", "phi", "normalize", "result", "best"]
        st.dataframe(df_sorted[display_cols], use_container_width=True)
    else:
        df_sorted = df.sort_values("test_mae", ascending=True).copy()
        df_sorted["best"] = ["*" if i == 0 else "" for i in range(len(df_sorted))]
        df_sorted["train_mae_fmt"] = df_sorted["train_mae"].apply(lambda x: f"{x:.4f}")
        df_sorted["test_mae_fmt"] = df_sorted["test_mae"].apply(lambda x: f"{x:.4f}")

        display_cols = ["D", "layers", "phi", "normalize", "pooling", "train_mae_fmt", "test_mae_fmt", "best"]
        st.dataframe(df_sorted[display_cols], use_container_width=True)


def render_chart(df: pd.DataFrame, batch_type: str) -> None:
    """Render Altair grouped bar chart."""
    if df.empty:
        st.info("No results match the current filters.")
        return

    df_chart = df.copy()

    if batch_type == "Node Classification":
        # Create unique config label including all params
        df_chart["config"] = df_chart.apply(
            lambda r: f"D={r['D']}, L={r['layers']}, {r['phi']}, {r['normalize']}", axis=1
        )
        # Sort for natural ordering: by D, layers, phi, normalize
        df_chart = df_chart.sort_values(["D", "layers", "phi", "normalize", "dataset"])
        config_order = df_chart["config"].unique().tolist()

        chart = alt.Chart(df_chart).mark_bar().encode(
            x=alt.X("config:N", title="Configuration", sort=config_order),
            y=alt.Y("accuracy:Q", title="Accuracy (%)"),
            color=alt.Color("dataset:N", title="Dataset"),
            xOffset="dataset:N",
            tooltip=["dataset", "D", "layers", "phi", "normalize", "accuracy", "std"]
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)
    else:
        # Create unique config label including all params
        df_chart["config"] = df_chart.apply(
            lambda r: f"D={r['D']}, L={r['layers']}, {r['phi']}, {r['normalize']}", axis=1
        )
        # Sort for natural ordering: by D, layers, phi, normalize
        df_chart = df_chart.sort_values(["D", "layers", "phi", "normalize", "pooling"])
        config_order = df_chart["config"].unique().tolist()

        chart = alt.Chart(df_chart).mark_bar().encode(
            x=alt.X("config:N", title="Configuration", sort=config_order),
            y=alt.Y("test_mae:Q", title="Test MAE"),
            color=alt.Color("pooling:N", title="Pooling"),
            xOffset="pooling:N",
            tooltip=["D", "layers", "phi", "normalize", "pooling", "train_mae", "test_mae"]
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)

st.title("GVFA Experiment Dashboard")

# Experiment selection
exp_type = st.selectbox(
    "Experiment Type",
    ["Node Classification", "Graph Regression (ZINC)"]
)

st.divider()

# Dataset selection (for node classification)
if exp_type == "Node Classification":
    datasets = st.multiselect(
        "Dataset",
        ["Cora", "citeseer", "Pubmed"],
        default=["Cora"]
    )

st.subheader("GVFA Parameters")

# Multi-select parameters
D_values = st.multiselect(
    "D (dimension)",
    [1000, 2000, 5000, 10000],
    default=[5000]
)

layers_values = st.multiselect(
    "num_layers",
    [2, 3, 4, 5],
    default=[3]
)

phi_values = st.multiselect(
    "phi",
    ["phi1", "phi2", "phi3", "phi4"],
    default=["phi3"]
)

norm_values = st.multiselect(
    "normalize",
    ["sign", "clip", "l2", "none"],
    default=["sign"]
)

# Experiment-specific parameters
st.divider()

if exp_type == "Node Classification":
    st.subheader("Node Classification Parameters")
    num_runs = st.slider("num_runs", min_value=1, max_value=20, value=5)

    # Build configurations
    if datasets and D_values and layers_values and phi_values and norm_values:
        configs = list(product(datasets, D_values, layers_values, phi_values, norm_values))
    else:
        configs = []
else:
    st.subheader("Graph Regression Parameters")
    col1, col2 = st.columns(2)
    with col1:
        pooling_values = st.multiselect(
            "pooling",
            ["sum", "mean"],
            default=["sum"]
        )
        alpha = st.selectbox("alpha", [0.1, 1.0, 10.0], index=1)
    with col2:
        seed = st.number_input("seed", min_value=0, value=42, step=1)

    # Build configurations
    if D_values and layers_values and phi_values and norm_values and pooling_values:
        configs = list(product(D_values, layers_values, phi_values, norm_values, pooling_values))
    else:
        configs = []

st.divider()

# Experiment count and validation
num_experiments = len(configs)

if num_experiments == 0:
    st.warning("Select at least one value for each parameter")
    can_run = False
elif num_experiments > 50:
    st.error(f"Too many experiments ({num_experiments}). Please reduce selections.")
    can_run = False
else:
    st.info(f"This will run {num_experiments} experiment(s)")
    can_run = True

# Run button
if st.button("Run All Experiments", type="primary", use_container_width=True, disabled=not can_run):
    # Import experiment modules only when needed
    from gvfa.experiments.node_classification_module import run_experiment as run_node_classification
    from gvfa.experiments.graph_regression_zinc import run_experiment as run_graph_regression

    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []

    if exp_type == "Node Classification":
        for i, (dataset, D, layers, phi, norm) in enumerate(configs):
            status_text.text(f"Running {dataset} with D={D}, layers={layers}...")
            mean, std = run_node_classification(
                dataset_name=dataset,
                D=D,
                num_layers=layers,
                phi=phi,
                normalize=norm,
                num_runs=num_runs
            )
            results.append({
                "dataset": dataset,
                "D": D,
                "layers": layers,
                "phi": phi,
                "normalize": norm,
                "accuracy": mean,
                "std": std,
                "result": f"{mean:.1f} +/- {std:.1f}%"
            })
            progress_bar.progress((i + 1) / num_experiments)

        status_text.text("All experiments completed!")

        # Store results in session state
        st.session_state.batch_results = results
        st.session_state.batch_type = "Node Classification"

        # Add to history
        if "experiment_history" not in st.session_state:
            st.session_state.experiment_history = []
        st.session_state.experiment_history.append({
            "id": len(st.session_state.experiment_history) + 1,
            "type": "Node Classification",
            "num_configs": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
    else:
        for i, (D, layers, phi, norm, pooling) in enumerate(configs):
            status_text.text(f"Running ZINC with D={D}, pooling={pooling}...")
            exp_results = run_graph_regression(
                D=D,
                num_layers=layers,
                phi=phi,
                normalize=norm,
                pooling=pooling,
                alpha=alpha,
                seed=int(seed)
            )
            results.append({
                "D": D,
                "layers": layers,
                "phi": phi,
                "normalize": norm,
                "pooling": pooling,
                "train_mae": exp_results['train_mae'],
                "test_mae": exp_results['test_mae']
            })
            progress_bar.progress((i + 1) / num_experiments)

        status_text.text("All experiments completed!")

        # Store results in session state
        st.session_state.batch_results = results
        st.session_state.batch_type = "Graph Regression"

        # Add to history
        if "experiment_history" not in st.session_state:
            st.session_state.experiment_history = []
        st.session_state.experiment_history.append({
            "id": len(st.session_state.experiment_history) + 1,
            "type": "Graph Regression",
            "num_configs": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        })

# Results display with tabs
st.divider()
st.subheader("Results")

if "batch_results" in st.session_state and st.session_state.batch_results:
    df = pd.DataFrame(st.session_state.batch_results)

    # Filters
    st.markdown("**Filters**")
    filtered_df = render_filters(df, st.session_state.batch_type, key_prefix="results")

    tab_table, tab_chart = st.tabs(["Table", "Chart"])

    with tab_table:
        render_table(filtered_df, st.session_state.batch_type)

    with tab_chart:
        render_chart(filtered_df, st.session_state.batch_type)
else:
    st.info("No experiments run yet. Results will appear here after running experiments.")

# History section
st.divider()
st.subheader("History")

if "experiment_history" in st.session_state and st.session_state.experiment_history:
    for batch in reversed(st.session_state.experiment_history):
        with st.expander(f"Batch #{batch['id']}: {batch['type']} ({batch['num_configs']} configs)"):
            batch_df = pd.DataFrame(batch["results"])

            st.markdown("**Filters**")
            filtered_batch_df = render_filters(batch_df, batch["type"], key_prefix=f"hist_{batch['id']}")

            hist_tab_table, hist_tab_chart = st.tabs(["Table", "Chart"])

            with hist_tab_table:
                render_table(filtered_batch_df, batch["type"])

            with hist_tab_chart:
                render_chart(filtered_batch_df, batch["type"])
else:
    st.info("No experiment history yet. Past experiments will appear here.")
