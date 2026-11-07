package predictor;

/**
 * One query against one synopsis. Features read their inputs from here; each shared intermediate result
 * (the query on the merged histogram, the query on every source, Skewness-Align) is computed on first use
 * and then reused, so a model only pays for the inputs it actually has.
 */
public class QueryFeatures {

	/** The query summed or weighted over the source histograms it overlaps (ml/features_v1.py source_features). */
	public static final class SourceAggregate {
		public double est, fullCount, width, contrast, inOut, empty, nPartial;
		public int nOverlap;
		public int resMin = 0, resMax = 0;
	}

	public final SynopsisContext ctx;
	public final double minLon, maxLon, minLat, maxLat;

	private HistogramQueryStats merged;
	private SourceAggregate sourceAgg;
	private Double skewPred;
	private Double mergeShift;

	public QueryFeatures(SynopsisContext ctx, double minLon, double maxLon, double minLat, double maxLat) {
		this.ctx = ctx;
		this.minLon = minLon;
		this.maxLon = maxLon;
		this.minLat = minLat;
		this.maxLat = maxLat;
	}

	public HistogramQueryStats merged() {
		if (merged == null) {
			merged = HistogramQueryStats.compute(ctx.merged, minLon, maxLon, minLat, maxLat);
		}
		return merged;
	}

	public double skewPred() {
		if (skewPred == null) {
			skewPred = ctx.skewnessAlign(minLon, maxLon, minLat, maxLat);
		}
		return skewPred;
	}

	/**
	 * (merged answer - sum of the source answers) / max(source sum, 1), from the histograms' own range-query
	 * answers. Training used those answers rather than the geometric estimate in {@link #merged()}; the two
	 * differ slightly on some queries, and the model was fitted to the former.
	 */
	public double mergeShift() {
		if (mergeShift == null) {
			double est = ctx.mergedHist.RangeQuery(minLon, maxLon, minLat, maxLat);
			double src = 0;
			for (histogram.SimpleSpatialHistogramOpt h : ctx.sourceHists) {
				src += h.RangeQuery(minLon, maxLon, minLat, maxLat);
			}
			mergeShift = (est - src) / Math.max(src, 1.0);
		}
		return mergeShift;
	}

	public SourceAggregate sources() {
		if (sourceAgg == null) {
			SourceAggregate s = new SourceAggregate();
			for (HistogramView h : ctx.sources) {
				if (!h.overlaps(minLon, maxLon, minLat, maxLat)) {
					continue;
				}
				HistogramQueryStats f = HistogramQueryStats.compute(h, minLon, maxLon, minLat, maxLat);
				s.est += f.est;
				s.fullCount += f.fullCount;
				s.width += f.touchedCount - f.fullCount;
				s.contrast += f.est * f.edgeContrast;
				s.inOut += f.est * f.inOut;
				s.empty += f.est * f.emptyPartial;
				s.nPartial += f.nPartial;
				int res = h.lonBuckets();
				s.resMin = s.nOverlap == 0 ? res : Math.min(s.resMin, res);
				s.resMax = s.nOverlap == 0 ? res : Math.max(s.resMax, res);
				s.nOverlap++;
			}
			sourceAgg = s;
		}
		return sourceAgg;
	}
}
