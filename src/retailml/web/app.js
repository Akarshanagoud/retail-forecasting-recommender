/* Demand Planning Studio: views over the retailml HTTP API. */
(function () {
  "use strict";
  const { h, api, fmt } = Kit;

  const VERDICT = { stable: { kind: "good", label: "Stable", color: "var(--good)" }, watch: { kind: "warning", label: "Watch", color: "var(--warning)" }, drifted: { kind: "critical", label: "Drifted", color: "var(--critical)" } };
  const REASON = {
    performance_drift: "forecast error rose beyond tolerance",
    feature_drift: "enough input features shifted",
    schedule: "the model is past its maximum age",
    no_production_model: "there is no production model",
  };
  const SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];

  let overviewCache = null;
  const overview = async () => overviewCache || (overviewCache = await api.get("/api/overview"));
  const modelLabel = (ov, name) => (ov.models.find((m) => m.name === name) || { label: name }).label;
  const seriesKey = (s) => s.sku + " @ " + s.store;
  const day = (iso) => new Date(iso + "T00:00:00").toLocaleDateString([], { month: "short", day: "numeric" });

  function seriesSelect(ov, sku, store, onChange) {
    return Kit.select(
      ov.series.map((s) => ({
        value: s.sku + "|" + s.store,
        label: seriesKey(s) + " · " + s.category + (s.intermittent ? " · intermittent" : "") + (s.regime_change_day ? " · demand shift" : ""),
      })),
      sku + "|" + store,
      (v) => onChange(...v.split("|")),
      { style: { width: "360px", maxWidth: "100%" }, "aria-label": "Series" },
    );
  }

  // ------------------------------------------------------------------ forecast
  async function forecastView(view, params) {
    const ov = await overview();
    const first = ov.series.find((s) => s.regime_change_day) || ov.series[0];
    const sku = params.sku || first.sku;
    const store = params.store || first.store;
    const model = params.model || "ridge";
    const d = await api.get("/api/series?sku=" + encodeURIComponent(sku) + "&store=" + encodeURIComponent(store) + "&model=" + model);
    const go = (patch) => app.go("forecast", Object.assign({ sku, store, model }, patch));

    const nHist = d.actual.length;
    const nAll = d.dates.length;
    const actual = d.dates.map((_, i) => (i < nHist ? d.actual[i] : null));
    const backtest = d.dates.map(() => null);
    d.windows.forEach((w) => w.predicted.forEach((p, j) => (backtest[w.start_index + j] = p)));
    const forecast = d.dates.map((_, i) => (i === nHist - 1 ? d.actual[i] : i >= nHist ? d.forecast[i - nHist] : null));
    const lo = d.dates.map((_, i) => (i >= nHist ? d.lower[i - nHist] : null));
    const hi = d.dates.map((_, i) => (i >= nHist ? d.upper[i - nHist] : null));
    const baseline = d.by_model.find((m) => m.model === "seasonal_naive");
    const m = d.metrics;

    view.append(
      Kit.pageHead("Forecast", "History, rolling-origin backtest forecasts and the next 28 days for one SKU at one store. The band is an 80% interval built from this model's own backtest errors."),
      h(
        "div",
        { class: "controls" },
        h("div", { class: "group" }, h("span", null, "Series"), seriesSelect(ov, sku, store, (a, b) => go({ sku: a, store: b }))),
        h("div", { class: "group" }, h("span", null, "Model"), Kit.select(ov.models.map((x) => ({ value: x.name, label: x.label })), model, (v) => go({ model: v }), { style: { width: "220px" }, "aria-label": "Model" })),
        d.spec.regime_change_day ? Kit.badge("Demand shifted on day " + d.spec.regime_change_day, "warning") : null,
        d.spec.intermittent ? Kit.badge("Intermittent demand", "info", { icon: false }) : null,
      ),
      h(
        "div",
        { class: "grid cols-5" },
        Kit.stat({ label: "Backtest WAPE", value: m.mean_wape === null ? "—" : fmt.pct(m.mean_wape, 1), note: baseline && baseline.mean_wape ? "seasonal naive " + fmt.pct(baseline.mean_wape, 1) : "" }),
        Kit.stat({ label: "MASE", value: fmt.num(m.mean_mase, 2), note: m.mean_mase !== null && m.mean_mase < 1 ? "beats same day last week" : "does not beat same day last week" }),
        Kit.stat({ label: "Spread across origins", value: m.wape_stdev === null ? "—" : fmt.pct(m.wape_stdev, 1), note: m.origins + " backtest origins" }),
        Kit.stat({ label: "Mean daily units", value: fmt.num(d.stats.mean_units, 1), note: fmt.pct(d.stats.zero_share, 1) + " zero-demand days" }),
        Kit.stat({ label: "Promotion days", value: fmt.int(d.stats.promotion_days), note: "of " + fmt.int(d.stats.days) + " days, price $" + fmt.num(d.spec.price, 2) }),
      ),
    );

    const chartCard = Kit.chartCard({
      title: seriesKey(d) + " · " + modelLabel(ov, model),
      subtitle: "Daily units. Dashed lines are forecasts; the shaded band is the 80% interval for the next 28 days.",
      legend: [
        { label: "Actual", color: "var(--series-1)", type: "line" },
        { label: "Backtest forecast", color: "var(--series-2)", type: "dash" },
        { label: "Forecast", color: "var(--series-3)", type: "dash" },
      ],
      chart: () =>
        Kit.line({
          labels: d.dates,
          series: [
            { name: "Actual", color: "var(--series-1)", values: actual, format: (v) => fmt.num(v, 0) + " units" },
            { name: "Backtest forecast", color: "var(--series-2)", values: backtest, dash: true, format: (v) => fmt.num(v, 0) + " units" },
            { name: "Forecast", color: "var(--series-3)", values: forecast, dash: true, format: (v) => fmt.num(v, 0) + " units" },
          ],
          band: { lo, hi, color: "var(--series-3)", label: "80% interval" },
          height: 300,
          yFormat: (v) => fmt.compact(v),
          xFormat: day,
          marker: nHist - 1,
          markerLabel: "forecast starts",
          extraRows: (i) => {
            const rows = [];
            if (i < nHist && d.promotion[i]) rows.push({ label: "On promotion", value: "yes" });
            if (i < nHist && d.holiday[i]) rows.push({ label: "Holiday", value: d.holiday[i] });
            return rows;
          },
          tooltipTitle: (iso, i) => new Date(iso + "T00:00:00").toLocaleDateString([], { weekday: "short", month: "short", day: "numeric", year: "numeric" }) + (i >= nHist ? " · forecast" : ""),
          ariaLabel: "Actual demand, backtest forecasts and the forward forecast",
        }),
      table: () => ({
        columns: [
          { key: "date", label: "Date" },
          { key: "actual", label: "Actual", align: "right", format: (v) => (v === null ? "—" : fmt.num(v, 0)) },
          { key: "backtest", label: "Backtest", align: "right", format: (v) => (v === null ? "—" : fmt.num(v, 1)) },
          { key: "forecast", label: "Forecast", align: "right", format: (v) => (v === null ? "—" : fmt.num(v, 1)) },
          { key: "range", label: "80% interval", align: "right" },
        ],
        rows: d.dates.map((iso, i) => ({ date: iso, actual: actual[i], backtest: backtest[i], forecast: i >= nHist ? forecast[i] : null, range: i >= nHist ? fmt.num(lo[i], 0) + "–" + fmt.num(hi[i], 0) : "—" })).reverse(),
      }),
    });

    const ranked = d.by_model.filter((x) => x.mean_wape !== null).sort((a, b) => a.mean_wape - b.mean_wape);
    const compareCard = Kit.chartCard({
      title: "Every model on this series",
      subtitle: "Backtest WAPE, lower is better. The selected model is highlighted.",
      chart: () =>
        Kit.hbars(
          ranked.map((x) => ({ label: x.label, value: x.mean_wape, color: x.model === model ? "var(--series-1)" : "var(--series-other)", extra: [{ label: "MASE", value: fmt.num(x.mean_mase, 2) }] })),
          { format: (v) => fmt.pct(v, 1), valueLabel: "WAPE", onClick: (x) => go({ model: ranked.find((r) => r.label === x.label).model }) },
        ),
      table: () => ({ columns: [{ key: "label", label: "Model" }, { key: "mean_wape", label: "WAPE", align: "right", format: (v) => fmt.pct(v, 1) }, { key: "mean_mase", label: "MASE", align: "right", format: (v) => fmt.num(v, 2) }], rows: ranked }),
    });
    const windowsCard = Kit.card({
      title: "Backtest windows on screen",
      subtitle: "Each is a refit on data before its cutoff, scored on the next 14 days",
      flush: true,
      body: Kit.table({
        columns: [
          { key: "cutoff", label: "Cutoff", render: (w) => day(w.cutoff) },
          { key: "wape", label: "WAPE", align: "right", format: (v) => (v === null ? "—" : fmt.pct(v, 1)) },
        ],
        rows: d.windows.slice().reverse(),
        emptyTitle: "No windows in view",
      }),
    });
    view.append(h("div", { class: "mt" }, chartCard), h("div", { class: "split-r mt" }, compareCard, windowsCard));
  }

  // ------------------------------------------------------------------ models
  function heatColor(v, lo, hi) {
    if (v === null || v === undefined) return { bg: "transparent", ink: "var(--muted)" };
    const t = hi > lo ? (v - lo) / (hi - lo) : 0;
    const i = Math.max(0, Math.min(SEQ.length - 1, Math.round(t * (SEQ.length - 1))));
    return { bg: SEQ[i], ink: i >= 6 ? "#ffffff" : "#0b0b0b" };
  }

  async function modelsView(view) {
    const ov = await overview();
    const data = await api.get("/api/models");
    const comps = data.comparisons;
    const best = comps[0];
    const baseline = comps.find((c) => c.model === "seasonal_naive");
    view.append(
      Kit.pageHead(
        "Model leaderboard",
        "Rolling-origin backtests: " + data.config.max_origins + " origins of " + data.config.horizon + " days across " + data.matrix.series.length + " series. Seasonal naive is the yardstick; MASE below 1 beats it.",
      ),
      h(
        "div",
        { class: "grid cols-4" },
        Kit.stat({ label: "Best model", value: best.label, note: "WAPE " + fmt.pct(best.wape, 1) }),
        Kit.stat({ label: "Improvement over baseline", value: baseline ? fmt.pct((baseline.wape - best.wape) / baseline.wape, 1) : "—", note: "relative WAPE reduction vs seasonal naive" }),
        Kit.stat({ label: "Beats the baseline on", value: best.beats_seasonal_naive + " windows", note: "backtest origins where it beat seasonal naive" }),
        Kit.stat({ label: "Series", value: fmt.int(ov.summary.series), note: fmt.int(ov.summary.rows) + " daily rows" }),
      ),
    );
    const bars = comps.map((c) => ({ label: c.label, value: c.wape, color: c.model === best.model ? "var(--series-1)" : "var(--series-other)", extra: [{ label: "MASE", value: fmt.num(c.mase, 3) }, { label: "Spread", value: fmt.pct(c.wape_stdev, 1) }] }));
    const lbCard = Kit.card({
      title: "Leaderboard",
      flush: true,
      body: Kit.table({
        columns: [
          { key: "label", label: "Model", render: (c) => h("span", null, c.label, c.model === "seasonal_naive" ? h("span", { class: "muted small" }, " · baseline") : null) },
          { key: "wape", label: "WAPE", align: "right", format: (v) => fmt.pct(v, 1) },
          { key: "wape_stdev", label: "Spread", align: "right", cls: "hide-sm", format: (v) => fmt.pct(v, 1) },
          { key: "mase", label: "MASE", align: "right", format: (v) => fmt.num(v, 3) },
          { key: "mase_badge", label: "vs baseline", sortable: false, render: (c) => (c.mase < 1 ? Kit.badge("Beats", "good") : Kit.badge("Worse", "critical")) },
          { key: "mean_fit_ms", label: "Fit time", align: "right", cls: "hide-sm", format: (v) => fmt.ms(v) },
        ],
        rows: comps,
        sort: { key: "wape", dir: "asc" },
      }),
    });
    const wapeCard = Kit.chartCard({
      title: "WAPE by model",
      subtitle: "Lower is better; the best model is highlighted",
      chart: () => Kit.hbars(bars, { format: (v) => fmt.pct(v, 1), valueLabel: "WAPE" }),
      table: () => ({ columns: [{ key: "label", label: "Model" }, { key: "value", label: "WAPE", align: "right", format: (v) => fmt.pct(v, 1) }], rows: bars }),
    });
    const all = data.matrix.wape.flat().filter((v) => v !== null).sort((a, b) => a - b);
    const lo = all[0];
    // Cap the scale at the 90th percentile: one pathological cell (a model blowing up on an
    // intermittent series) would otherwise flatten every other cell to the lightest step.
    const hi = all[Math.floor(0.9 * (all.length - 1))];
    const heat = h(
      "div",
      { class: "table-wrap" },
      h(
        "table",
        { class: "tbl heat" },
        h("thead", null, h("tr", null, h("th", null, "Series"), data.matrix.models.map((mname) => h("th", { class: "rot r" }, modelLabel(ov, mname))))),
        h(
          "tbody",
          null,
          data.matrix.series.map((label, r) => {
            const [skuPart, storePart] = label.split(" @ ");
            const tr = h("tr", { class: "clickable", tabindex: "0" }, h("td", null, label));
            data.matrix.wape[r].forEach((v, c) => {
              const col = heatColor(v, lo, hi);
              const td = h("td", { class: "cell", style: { background: col.bg, color: col.ink } }, v === null ? "—" : fmt.pct(v, 1));
              Kit.attachTip(td, () => Kit.ttContent(label, [{ label: modelLabel(ov, data.matrix.models[c]), value: v === null ? "n/a" : fmt.pct(v, 1) }]));
              tr.append(td);
            });
            tr.addEventListener("click", () => app.go("forecast", { sku: skuPart, store: storePart, model: best.model }));
            return tr;
          }),
        ),
      ),
    );
    view.append(
      h("div", { class: "split-r mt" }, lbCard, wapeCard),
      Kit.card({ cls: "mt", title: "WAPE by series and model", subtitle: "Darker means more error; the darkest step is " + fmt.pct(hi, 0) + " and above. Intermittent and shifted series stand out. Click a row to open its forecast.", actions: h("span", { class: "scale" }, fmt.pct(lo, 0), h("i"), fmt.pct(hi, 0) + "+"), flush: true, body: heat }),
    );
  }

  // ------------------------------------------------------------------ recommendations
  async function recommendView(view, params) {
    const customers = await api.get("/api/customers");
    let current = params.customer || customers[0].customer;
    const list = h("div", { class: "list" });
    const detail = h("div", null);
    const search = h("input", { class: "input", type: "search", placeholder: "Find a customer", "aria-label": "Find a customer" });

    const drawList = (rows) => {
      Kit.clear(list);
      rows.forEach((c) => {
        const item = h("div", { class: "list-item clickable" + (c.customer === current ? " selected" : ""), tabindex: "0" }, h("div", { style: { flex: "1" } }, h("b", null, c.customer), h("div", { class: "small muted" }, c.purchases + " units · " + c.distinct + " products")));
        item.addEventListener("click", () => {
          current = c.customer;
          list.querySelectorAll(".list-item").forEach((x) => x.classList.remove("selected"));
          item.classList.add("selected");
          show(c.customer);
        });
        list.append(item);
      });
    };
    search.addEventListener("input", async () => drawList(await api.get("/api/customers?q=" + encodeURIComponent(search.value))));

    const show = async (customer) => {
      const r = await Kit.busy(detail, () => api.get("/api/recommend?customer=" + encodeURIComponent(customer)));
      Kit.clear(detail);
      const maxScore = Math.max(...r.cf.map((x) => x.score), 1e-9);
      detail.append(
        Kit.card({
          title: customer,
          subtitle: "Purchase history in the training window",
          body: h("div", { class: "row", style: { gap: "6px" } }, r.history.map((x) => h("span", { class: "sku-chip" }, h("b", null, x.sku), h("span", { class: "muted" }, x.category + " ×" + x.quantity)))),
        }),
        h(
          "div",
          { class: "split-r mt" },
          Kit.card({
            title: "Recommended for this customer",
            subtitle: "Item-item collaborative filtering, explained by the purchases that produced each suggestion",
            flush: true,
            body: Kit.table({
              columns: [
                { key: "rank", label: "#", align: "right", width: "36px", render: (x) => String(r.cf.indexOf(x) + 1) },
                { key: "sku", label: "Product", render: (x) => h("div", null, h("b", { class: "nowrap" }, x.sku), h("div", { class: "small muted" }, x.category)) },
                { key: "score", label: "Score", render: (x) => h("span", { class: "row", style: { flexWrap: "nowrap" } }, h("span", { style: { width: "70px" } }, Kit.meter(x.score / maxScore, "", "Score")), h("span", { class: "num small" }, fmt.num(x.score, 2))) },
                { key: "because_of", label: "Because they bought", sortable: false, render: (x) => h("span", { class: "small" }, x.because_of.map((b, i) => [i ? ", " : null, h("span", { class: "nowrap" }, b)])) },
              ],
              rows: r.cf,
            }),
          }),
          Kit.card({
            title: "Bestsellers baseline",
            subtitle: "What everyone would see without personalisation",
            flush: true,
            body: Kit.table({ columns: [{ key: "sku", label: "Product", render: (x) => h("b", null, x.sku) }, { key: "category", label: "Category" }], rows: r.popular }),
          }),
        ),
      );
      const metrics = [
        ["precision@10", "Precision"],
        ["recall@10", "Recall"],
        ["map@10", "MAP"],
        ["ndcg@10", "nDCG"],
        ["hit_rate", "Hit rate"],
        ["catalogue_coverage", "Coverage"],
      ];
      const cf = r.scores["item-item CF"];
      const pop = r.scores.popularity;
      detail.append(
        h(
          "div",
          { class: "mt" },
          Kit.chartCard({
            title: "Offline evaluation",
            subtitle: "Temporal split: trained on the past, scored on each customer's later purchases (k = 10). Novelty: CF " + fmt.num(cf.novelty, 2) + ", popularity " + fmt.num(pop.novelty, 2) + ".",
            legend: [
              { label: "Item-item CF", color: "var(--series-1)" },
              { label: "Popularity baseline", color: "var(--series-other)" },
            ],
            chart: () =>
              Kit.columns({
                labels: metrics.map((x) => x[1]),
                series: [
                  { name: "Item-item CF", color: "var(--series-1)", values: metrics.map((x) => cf[x[0]]), format: (v) => fmt.num(v, 3) },
                  { name: "Popularity baseline", color: "var(--series-other)", values: metrics.map((x) => pop[x[0]]), format: (v) => fmt.num(v, 3) },
                ],
                stacked: false,
                height: 230,
                yMax: 1,
                yFormat: (v) => v.toFixed(2),
                ariaLabel: "Recommendation metrics, collaborative filtering against the popularity baseline",
              }),
            table: () => ({
              columns: [{ key: "metric", label: "Metric" }, { key: "cf", label: "Item-item CF", align: "right", format: (v) => fmt.num(v, 3) }, { key: "pop", label: "Popularity", align: "right", format: (v) => fmt.num(v, 3) }],
              rows: metrics.map(([k, label]) => ({ metric: label, cf: cf[k], pop: pop[k] })).concat([{ metric: "Novelty", cf: cf.novelty, pop: pop.novelty }]),
            }),
          }),
        ),
      );
    };

    view.append(
      Kit.pageHead("Recommendations", "Explainable item-item collaborative filtering, measured against simply showing everyone the bestsellers."),
      h("div", { class: "split" }, Kit.card({ title: "Customers", subtitle: "Most active first", body: h("div", { class: "stack", style: { gap: "10px" } }, search, h("div", { class: "scroll-y", style: { maxHeight: "640px", margin: "0 -18px" } }, list)) }), detail),
    );
    drawList(customers);
    show(current);
  }

  // ------------------------------------------------------------------ drift
  async function driftView(view, params) {
    const ov = await overview();
    const first = ov.series.find((s) => s.regime_change_day) || ov.series[0];
    const sku = params.sku || first.sku;
    const store = params.store || first.store;
    const age = Number(params.age || 12);
    const d = await api.get("/api/drift?sku=" + encodeURIComponent(sku) + "&store=" + encodeURIComponent(store) + "&model_age_days=" + age);
    const go = (patch) => app.go("drift", Object.assign({ sku, store, age }, patch));
    const rep = d.report;
    const dec = d.decision;
    view.append(
      Kit.pageHead("Drift and retraining", "Drift is evidence, not a trigger by itself: retraining needs performance drift, enough shifted features, or an old model. Promotion then needs a material improvement."),
      h(
        "div",
        { class: "controls" },
        h("div", { class: "group" }, h("span", null, "Series"), seriesSelect(ov, sku, store, (a, b) => go({ sku: a, store: b }))),
        h("div", { class: "group" }, h("span", null, "Production model age"), Kit.select([3, 12, 30, 45, 90].map((v) => ({ value: v, label: v + " days" })), age, (v) => go({ age: Number(v) }), { style: { width: "110px" }, "aria-label": "Model age" })),
      ),
      dec.should_retrain
        ? Kit.banner({ kind: "warning", title: "Retraining recommended", text: dec.reasons.map((r) => REASON[r] || r).join("; ") + ". A candidate still has to beat production by 2% to be promoted.", extra: h("div", { style: { marginTop: "8px" } }, h("a", { class: "btn btn-sm", href: "#/registry" }, "Open the model registry")) })
        : Kit.banner({ kind: "good", title: "No retraining needed", text: dec.detail.feature_drift_below_threshold ? "Some features moved (" + dec.detail.feature_drift_below_threshold + "), but not enough to act on." : "Inputs and error are within tolerance." }),
    );
    const perf = rep.performance;
    view.append(
      h(
        "div",
        { class: "grid cols-4 mt" },
        Kit.stat({ label: "Overall verdict", value: (VERDICT[rep.verdict] || { label: rep.verdict }).label, note: rep.drifted_features.length + " of " + rep.features.length + " features drifted" }),
        Kit.stat({ label: "Earlier backtest WAPE", value: fmt.pct(d.baseline_wape, 1), note: "first half of the origins" }),
        Kit.stat({ label: "Recent backtest WAPE", value: fmt.pct(d.current_wape, 1), note: perf ? (perf.statistic >= 0 ? "+" : "") + fmt.pct(perf.statistic, 1) + " relative change" : "" }),
        Kit.stat({ label: "Split date", value: day(d.current_period[0]), note: "reference from " + day(d.reference_period[0]) + ", current to " + day(d.current_period[1]) }),
      ),
    );
    const feats = rep.features.slice().sort((a, b) => b.statistic - a.statistic);
    view.append(
      h(
        "div",
        { class: "split-r mt" },
        Kit.chartCard({
          title: "Feature drift (PSI)",
          subtitle: "Population stability index, reference half against current half. Coloured by verdict.",
          legend: ["stable", "watch", "drifted"].map((v) => ({ label: VERDICT[v].label, color: VERDICT[v].color })),
          chart: () => Kit.hbars(feats.slice(0, 16).map((f) => ({ label: f.feature, value: f.statistic, color: (VERDICT[f.verdict] || {}).color, extra: [{ label: "KS", value: fmt.num(f.detail.ks, 3) }, { label: "Verdict", value: (VERDICT[f.verdict] || { label: f.verdict }).label }] })), { format: (v) => fmt.num(v, 3), valueLabel: "PSI" }),
          table: () => ({ columns: [{ key: "feature", label: "Feature" }, { key: "statistic", label: "PSI", align: "right", format: (v) => fmt.num(v, 4) }, { key: "verdict", label: "Verdict" }], rows: feats }),
        }),
        Kit.card({
          title: "All features",
          flush: true,
          body: Kit.table({
            columns: [
              { key: "feature", label: "Feature", render: (f) => h("code", null, f.feature) },
              { key: "statistic", label: "PSI", align: "right", format: (v) => fmt.num(v, 3) },
              { key: "ks", label: "KS", align: "right", cls: "hide-sm", render: (f) => fmt.num(f.detail.ks, 3), sortValue: (f) => f.detail.ks },
              { key: "verdict", label: "Verdict", render: (f) => Kit.badge((VERDICT[f.verdict] || { label: f.verdict }).label, (VERDICT[f.verdict] || {}).kind) },
            ],
            rows: feats,
            maxHeight: 520,
          }),
        }),
      ),
    );
  }

  // ------------------------------------------------------------------ registry
  async function registryView(view) {
    const ov = await overview();
    let reg = await api.get("/api/registry");
    const host = h("div", null);
    const modelSel = Kit.select(ov.models.map((m) => ({ value: m.name, label: m.label })), "ridge", null, { style: { width: "220px" }, "aria-label": "Model to train" });
    const trainBtn = h("button", { class: "btn btn-primary", type: "button" }, "Train candidate");
    trainBtn.addEventListener("click", () =>
      Kit.withButton(trainBtn, async () => {
        const r = await api.post("/api/registry/train", { model: modelSel.value });
        reg = r.registry;
        Kit.toast("Registered v" + r.registered.version + " (" + reg.labels[r.registered.params.model] + ") in staging");
        draw();
      }),
    );
    const draw = () => {
      Kit.clear(host);
      const prod = reg.production;
      host.append(
        prod
          ? Kit.banner({ kind: "good", title: "In production: v" + prod.version + " · " + reg.labels[prod.params.model], text: "Backtest WAPE " + fmt.pct(prod.metric.wape, 2) + " · registered " + new Date(prod.registered_at).toLocaleString() + " · run " + prod.run_id })
          : Kit.banner({ kind: "warning", title: "No production model" }),
        h(
          "div",
          { class: "split-r mt" },
          Kit.card({
            title: "Versions",
            subtitle: "A staging version is promoted only if it improves backtest WAPE by at least 2%",
            flush: true,
            body: Kit.table({
              columns: [
                { key: "version", label: "Version", render: (v) => h("b", null, "v" + v.version) },
                { key: "model", label: "Model", render: (v) => reg.labels[v.params.model] || v.params.model, sortable: false },
                { key: "wape", label: "WAPE", align: "right", render: (v) => fmt.pct(v.metric.wape, 2), sortValue: (v) => v.metric.wape },
                { key: "stage", label: "Stage", render: (v) => Kit.badge(fmt.title(v.stage), v.stage === "production" ? "good" : v.stage === "archived" ? "" : "info", { icon: v.stage === "production" }) },
                {
                  key: "action",
                  label: "",
                  sortable: false,
                  align: "right",
                  render: (v) => {
                    if (v.stage !== "staging") return null;
                    const b = h("button", { class: "btn btn-sm", type: "button" }, "Promote");
                    b.addEventListener("click", () =>
                      Kit.withButton(b, async () => {
                        const r = await api.post("/api/registry/promote", { version: v.version });
                        reg = r.registry;
                        Kit.toast((r.promoted ? "Promoted: " : "Not promoted: ") + r.reason, r.promoted ? "" : "error");
                        draw();
                      }),
                    );
                    return b;
                  },
                },
              ],
              rows: reg.versions,
            }),
          }),
          Kit.card({
            title: "Train a candidate",
            subtitle: "Logs a tracked run from the backtest and registers it in staging",
            body: h("div", { class: "stack" }, h("div", { class: "row" }, modelSel, trainBtn), h("p", { class: "small muted" }, "Try Ridge regression first (a large improvement over the baseline), then Ensemble: the gate refuses a candidate that is not materially better than production.")),
          }),
        ),
        Kit.card({
          cls: "mt",
          title: "Experiment runs",
          subtitle: "Every training run with its parameters, metrics and data fingerprint",
          flush: true,
          body: Kit.table({
            columns: [
              { key: "run_id", label: "Run", render: (r) => h("code", null, r.run_id) },
              { key: "model", label: "Model", render: (r) => reg.labels[r.params.model] || r.params.model, sortable: false },
              { key: "wape", label: "WAPE", align: "right", render: (r) => fmt.pct(r.metrics.wape, 2), sortValue: (r) => r.metrics.wape },
              { key: "mase", label: "MASE", align: "right", cls: "hide-sm", render: (r) => fmt.num(r.metrics.mase, 3), sortValue: (r) => r.metrics.mase },
              { key: "status", label: "Status", cls: "hide-sm", render: (r) => fmt.title(r.status || "finished") },
              { key: "started_at", label: "Started", cls: "hide-sm", render: (r) => new Date(r.started_at).toLocaleTimeString() },
              { key: "git_commit", label: "Commit", cls: "hide-sm", render: (r) => ((r.environment || {}).git_commit ? h("code", null, String(r.environment.git_commit).slice(0, 7)) : "—") },
            ],
            rows: reg.runs,
          }),
        }),
      );
    };
    view.append(Kit.pageHead("Model registry", "Versioned forecasting models with gated promotion, as the daily Airflow DAG runs it."), host);
    draw();
  }

  // ------------------------------------------------------------------ boot
  let app = null;
  async function boot() {
    const outlet = document.getElementById("app");
    const pill = document.getElementById("env");
    Kit.initTheme(document.getElementById("theme-toggle"));
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    for (;;) {
      let info = null;
      try {
        info = await api.get("/health");
      } catch (_) {
        await sleep(1500);
        continue;
      }
      if (info.status === "ok") break;
      if (info.status === "error") {
        Kit.clear(outlet).append(Kit.errorState(new Error(info.error || "start-up failed")));
        return;
      }
      const p = info.progress;
      pill.textContent = "warming up";
      Kit.clear(outlet).append(
        h(
          "div",
          { class: "progress-wrap" },
          Kit.card({
            title: "Running backtests",
            subtitle: "Every model on every series, refit at each origin. This happens once, at start-up.",
            body: h("div", { class: "stack", style: { gap: "8px" } }, Kit.meter(p.done / p.total, "", "Progress"), h("div", { class: "row-between small" }, h("span", { class: "ink-2" }, fmt.title(p.stage)), h("span", { class: "num muted" }, p.done + " / " + p.total))),
          }),
        ),
      );
      await sleep(800);
    }
    app = Kit.createApp({
      title: "Demand Planning Studio",
      health: "/health",
      healthLabel: (info) => info.series + " series · ready in " + info.startup_seconds + " s",
      routes: [
        { path: "forecast", label: "Forecast", render: forecastView },
        { path: "models", label: "Models", render: modelsView },
        { path: "recommend", label: "Recommendations", render: recommendView },
        { path: "drift", label: "Drift", render: driftView },
        { path: "registry", label: "Registry", render: registryView },
      ],
    });
  }
  boot();
})();
