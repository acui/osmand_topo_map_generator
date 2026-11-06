package net.osmand.plus.plugins.topoprint.render;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.os.Handler;
import android.util.Log;
import android.os.Looper;
import android.os.SystemClock;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import net.osmand.core.android.AtlasMapRendererView;
import net.osmand.core.android.MapRendererContext;
import net.osmand.core.android.MapRendererView;
import net.osmand.core.android.MapRendererView.MapRendererViewListener;
import net.osmand.core.jni.PointI;
import net.osmand.core.jni.ZoomLevel;
import net.osmand.data.LatLon;
import net.osmand.plus.OsmandApplication;
import net.osmand.plus.R;
import net.osmand.plus.plugins.PluginsHelper;
import net.osmand.plus.plugins.topoprint.PrintParameters;
import net.osmand.plus.plugins.topoprint.TopoPrintPlugin;
import net.osmand.plus.plugins.topoprint.compose.UtmMath;
import net.osmand.plus.utils.NativeUtilities;
import net.osmand.plus.views.OsmandMapTileView;
import net.osmand.plus.views.corenative.NativeCoreContext;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * Renders the print basemap through the OpenGL engine, rebinding the global
 * MapRendererContext to a dedicated full-resolution view (Android Auto
 * SurfaceRenderer pattern). Spherical projection is supported via setFlatEarth.
 *
 * Threading: MapRendererView is a ViewGroup, so every lifecycle call must run
 * on the main thread — the flow is split into small main-thread steps. Only
 * the frame wait runs on a worker thread. Result is delivered on main.
 */
public class GLOffscreenRenderer implements MapRendererViewListener {

	private static final String TAG = "TopoPrintGL";
	private static final long FRAME_WAIT_TIMEOUT_MS = 300000;
	private static final long FRAME_POLL_MS = 250;
	private static final long FRAME_STABLE_MS = 15000;
	private static final long MIN_SOAK_MS = 12000;
	/**
	 * Residual zoom error the calibration accepts: the measured ground
	 * coverage must match the requested frame within 0.2% (≈0.4mm on a 175mm
	 * paper frame at 1:25000), otherwise the printed scale drifts visibly
	 * against the printed kilometre grid.
	 */
	private static final double ZOOM_TOLERANCE_RATIO = 0.002;
	private static final double CENTER_TOLERANCE_MIN_METERS = 5.0;

	/**
	 * Centre tolerance, scaled with the printed coverage (0.2%, ≥5 m). A
	 * fixed 5 m rejected large-format maps over systematic sub-millimetre
	 * offsets and provoked needless target nudges that flipped the camera
	 * into its alternate state on spherical big viewports.
	 */
	private double centerToleranceMeters() {
		return Math.max(CENTER_TOLERANCE_MIN_METERS,
				0.002 * Math.max(widthMeters, heightMeters));
	}
	private static final int CAPTURE_RETRIES = 8;
	private static final int CAPTURE_BURSTS = 4;
	private static final int DRIFT_SAMPLES = 5;
	private static final long DRIFT_SAMPLE_INTERVAL_MS = 60;
	private static final long CAPTURE_RETRY_DELAY_MS = 2000;

	private final long startUptimeMs = SystemClock.uptimeMillis();

	private final OsmandApplication app;
	private final PrintParameters params;
	private final LatLon center;
	private final double widthMeters;
	private final double heightMeters;
	private final boolean spherical;
	@Nullable
	private final OffscreenMapRenderer.ProgressListener progress;
	private final OffscreenMapRenderer.RenderListener listener;
	private final Handler mainHandler = new Handler(Looper.getMainLooper());

	private MapRendererContext savedContext;
	private float savedDensity;
	private MapRendererView mainRendererView;
	private MapRendererContext printContext;
	private AtlasMapRendererView offscreenView;
	private OsmandMapTileView tileView;
	/**
	 * True once {@link #offscreenView} completed setupRenderer and got this
	 * listener attached. MapRendererView.removeListener() synchronizes on its
	 * _mapRenderer field, which is null while setupRenderer has not (or not
	 * successfully) run — calling it on a failed-setup view crashes with
	 * "Null reference used for synchronization".
	 */
	private boolean listenerAttached;
	/** Last fractional zoom passed to applyFractionalZoom — the calibration's reference. */
	private double lastTargetFractionalZoom;
	private volatile boolean frameReady;
	private volatile boolean delivered;

	public GLOffscreenRenderer(@NonNull OsmandApplication app, @NonNull PrintParameters params,
	                           @NonNull LatLon center, double widthMeters, double heightMeters,
	                           boolean spherical, @Nullable OffscreenMapRenderer.ProgressListener progress,
	                           @NonNull OffscreenMapRenderer.RenderListener listener) {
		this.app = app;
		this.params = params;
		this.center = center;
		this.widthMeters = widthMeters;
		this.heightMeters = heightMeters;
		this.spherical = spherical;
		this.progress = progress;
		this.listener = listener;
	}

	public void start() {
		reportProgress(R.string.topoprint_progress_prepare);
		app.runInUIThread(this::step1CreateContext);
	}

	/** Stage reporter; safe from any thread, delivered on the UI thread. */
	private void reportProgress(int resId, @Nullable Object... args) {
		OffscreenMapRenderer.ProgressListener listener = progress;
		if (listener == null) return;
		String message = args.length == 0
				? app.getString(resId) : app.getString(resId, args);
		app.runInUIThread(() -> listener.onProgress(message));
	}

	private void step1CreateContext() {
		if (delivered) return;
		// Previous renders leave strings, buffers and cached objects behind;
		// collect them before the big allocations of this render.
		System.gc();
		tileView = app.getOsmandMap().getMapView();
		mainRendererView = tileView.getMapRenderer();
		savedContext = NativeCoreContext.getMapRendererContext();
		savedDensity = savedContext != null ? savedContext.getDensity() : 1.0f;
		Log.i(TAG, "render start: rasterPx=" + rasterWidthPx() + "x" + rasterHeightPx()
				+ " coverageMeters=" + widthMeters + "x" + heightMeters
				+ " spherical=" + spherical + " center=" + center);
		if (!OffscreenMapRenderer.isBitmapSizeAllowed(app, rasterWidthPx(), rasterHeightPx())) {
			finish(null, 0, 0);
			return;
		}
		// Same handover pattern as the Android Auto SurfaceRenderer: unbind the
		// main view from its context, replace the global context with a print
		// one, and let setupRenderer() hand the live EGL renderer over.
		if (savedContext != null && savedContext.getMapRendererView() != null) {
			savedContext.setMapRendererView(null);
		}
		float supersampling = (float) OffscreenMapRenderer.renderDpi(params) / (float) OffscreenMapRenderer.STYLE_DPI;
		// Print presentation (no user-setting mutation): mapScaleFactor 1.0 —
		// identical to the EyePiece/map-rasterizer reference pipeline, so road
		// stroke widths stay proportional to areas/buildings. Label detail is
		// NOT a presentation-density matter: styles gate street names, POI
		// icons and road geometry on the INTEGER zoom level the core splits
		// the fractional camera zoom into (round(zoom), see
		// MapRenderer::setZoomToState). The map camera therefore pins its
		// detail level via applyFractionalZoom() below.
		NativeCoreContext.setMapRendererContext(app, supersampling);
		printContext = NativeCoreContext.getMapRendererContext();
		printContext.setMapDensityOverride(1.0f);
		// Map label knob: overrides TEXT_SCALE (symbolsScaleFactor) for the
		// print presentation, so label size — and through collision avoidance
		// the number of labels that fit — can be tuned independently of the
		// app's map text setting. 1.0 keeps the override off.
		float symbolsScale = PluginsHelper.getPlugin(TopoPrintPlugin.class).getSymbolsScale();
		if (Math.abs(symbolsScale - 1.0f) > 0.01f) {
			printContext.setSymbolsScaleOverride(symbolsScale);
		}
		// The context is reused when the supersampling matches a previous
		// print render — force the presentation rebuild so the overrides are
		// actually applied.
		printContext.updateMapSettings(true);
		Log.i(TAG, "print presentation: rasterMapScale=1.0 symbolsScale=" + symbolsScale
				+ " (style: " + app.getSettings().RENDERER.get() + ")");
		// Detach the main map: freezes it during the print render and avoids
		// two GL contexts fighting over the (software) renderer.
		tileView.setMapRenderer(null, true);
		mainHandler.post(this::step2SetupRenderer);
	}

	private void step2SetupRenderer() {
		if (delivered) return;
		reportProgress(R.string.topoprint_progress_gl);
		offscreenView = new AtlasMapRendererView(app);
		printContext.presetMapRendererOptions(offscreenView, false);
		try {
			offscreenView.setupRenderer(app, rasterWidthPx(), rasterHeightPx(), mainRendererView);
		} catch (OutOfMemoryError e) {
			// setupRenderer allocates the frame read-back buffer (W×H×4
			// bytes) as a Java direct buffer, which counts against the heap
			// growth limit. On huge maps (A3@600dpi ≈ 180 MB) leftovers
			// from previous renders can crowd the heap — force a collection
			// and retry once before failing with the lower-DPI hint.
			Log.w(TAG, "setupRenderer OOM — forcing GC and retrying once", e);
			System.gc();
			SystemClock.sleep(500);
			try {
				offscreenView.setupRenderer(app, rasterWidthPx(), rasterHeightPx(), mainRendererView);
			} catch (OutOfMemoryError e2) {
				Log.e(TAG, "setupRenderer OOM after retry — bitmap " + rasterWidthPx() + "x"
						+ rasterHeightPx() + " does not fit the heap", e2);
				finish(null, 0, 0);
				return;
			}
		}
		Log.i(TAG, "renderer setup done: viewPx=" + rasterWidthPx() + "x" + rasterHeightPx());
		offscreenView.setMinZoomLevel(ZoomLevel.swigToEnum(4));
		offscreenView.setMaxZoomLevel(ZoomLevel.swigToEnum(22));
		offscreenView.setAzimuth(0);
		offscreenView.setFlatEarth(!spherical);
		// Re-bind view-attached raster layers (terrain/hillshade, overlays) to
		// the offscreen renderer: their providers are registered per
		// MapRendererView and would otherwise be missing from the map. The
		// tileView is frozen during the print, so force one synchronous draw
		// pass to run the deferred provider rebinds here and now.
		tileView.setMapRenderer(offscreenView, true);
		Bitmap probe = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888);
		Canvas probeCanvas = new Canvas(probe);
		tileView.drawOverMap(probeCanvas, tileView.getCurrentRotatedTileBox(),
				new net.osmand.plus.views.layers.base.OsmandMapLayer.DrawSettings(false, true));
		probeCanvas.setBitmap(null);
		probe.recycle();
		// Layer-owned symbol providers (location, markers) are stripped after
		// the draw pass - only the map and its raster layers belong in the
		// print.
		offscreenView.removeAllSymbolsProviders();
		offscreenView.resumeSymbolsUpdate();
		offscreenView.setSymbolsUpdateInterval(1000);
		offscreenView.setMaximumFrameRate(30);
		offscreenView.addListener(this);
		listenerAttached = true;
		// Without the frame reader getBitmap() returns the raw GL frame whose
		// rows are bottom-up (vertically flipped relative to screen space).
		offscreenView.enableFrameReader();
		printContext.setMapRendererView(offscreenView);
		mainHandler.post(this::step3ConfigureTarget);
	}

	private void step3ConfigureTarget() {
		if (delivered || offscreenView == null) return;
		// Style-anchor fractional zoom (300 dpi reference, identical to the
		// EyePiece CLI). The render-DPI formula (groundMetersPerPixel) reads
		// one level higher at 600 dpi; combined with round-to-nearest that
		// yields the same camera, but with ceil it pushed the detail level
		// to z17/z19 where the symbol payload overwhelms large viewports.
		double zoomFractional = OffscreenMapRenderer.fractionalZoom(params, center.getLatitude());
		applyFractionalZoom(zoomFractional);
		offscreenView.setMapTargetLocation(
				NativeUtilities.getPoint31FromLatLon(center.getLatitude(), center.getLongitude()));
		Log.i(TAG, "target set: zoomFractional=" + zoomFractional + " center=" + center);
		frameReady = false;
		Thread waiter = new Thread(this::waitForFrames, "TopoPrintFrameWait");
		waiter.start();
	}

	private void waitForFrames() {
		reportProgress(R.string.topoprint_progress_tiles);
		long deadline = SystemClock.uptimeMillis() + FRAME_WAIT_TIMEOUT_MS;
		// 1. Wait for the first rendered frame.
		while (!frameReady && SystemClock.uptimeMillis() < deadline) {
			SystemClock.sleep(FRAME_POLL_MS);
		}
		Log.i(TAG, "first frame: frameReady=" + frameReady
				+ " afterMs=" + (SystemClock.uptimeMillis() - startUptimeMs));
		// 2. Map tiles load asynchronously — wait until the frame id stops
		//    advancing (data settled), with a minimum soak time.
		waitForStable(deadline);
		// 3. Calibrate the zoom against the *actual* on-screen coverage: the
		//    renderer's zoom semantics differ between Mercator and spherical
		//    projections, so instead of trusting the formula, measure the
		//    ground extent through the renderer's own screen→location
		//    raycast and correct the zoom until the coverage matches the
		//    requested frame (≤1%).
		double finalRatio = 1.0;
		for (int round = 0; round < 3; round++) {
			Double ratio = measureCoverageRatioOnUi();
			Log.i(TAG, "zoom calibration round " + round + ": ratio=" + ratio);
			if (ratio == null || Math.abs(ratio - 1.0) <= ZOOM_TOLERANCE_RATIO) {
				break;
			}
			// Ground extent ∝ 2^(−zoom): if the measured coverage is ratio×
			// the requested one, zoom in by log2(ratio) to shrink it back.
			adjustZoomOnUiThread(Math.log(ratio) / Math.log(2));
			frameReady = false;
			waitForStable(deadline);
			Double check = measureCoverageRatioOnUi();
			finalRatio = check == null ? Double.NaN : check;
			Log.i(TAG, "zoom calibration round " + round + " verified ratio=" + check);
		}
		// 4. Recenter: in the spherical projection the renderer's target→
		//    view mapping deviates from the request, so raycast the viewport
		//    centre and slide the target by the observed 31-space offset until
		//    the requested centre really sits at the middle of the raster.
		for (int round = 0; round < 3; round++) {
			double driftMeters = calibrateCenterOnUi();
			Log.i(TAG, "center calibration round " + round + ": driftMeters=" + driftMeters);
			if (driftMeters <= centerToleranceMeters()) {
				break;
			}
			frameReady = false;
			waitForStable(deadline);
		}
		// 5. Capture is gated inside captureNow: the renderer has been observed
		//    to alternate between two camera states (the corrected one and one
		//    ~1.5 km off) on consecutive raycasts, so the gate samples several
		//    times and only captures in a verified window.
		final double verifiedRatio = finalRatio;
		Log.i(TAG, "calibration finished: verifiedRatio=" + verifiedRatio);
		app.runInUIThread(() -> captureNow(0));
	}

	/** Runs on the worker thread; returns the remaining drift in meters. */
	private double calibrateCenterOnUi() {
		final CountDownLatch latch = new CountDownLatch(1);
		final double[] drift = {Double.MAX_VALUE};
		mainHandler.post(() -> {
			drift[0] = recenterTarget();
			latch.countDown();
		});
		try {
			latch.await(3, TimeUnit.SECONDS);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
		return drift[0];
	}

	/** Runs on the UI thread; slides the target by the measured 31 offset. */
	private double recenterTarget() {
		if (offscreenView == null) {
			return 0;
		}
		int w = rasterWidthPx();
		int h = rasterHeightPx();
		LatLon actual = NativeUtilities.getLatLonFromPixel(offscreenView, null, w / 2, h / 2);
		if (actual == null) {
			Log.w(TAG, "recenter: centre raycast failed");
			return 0;
		}
		PointI actual31 = NativeUtilities.getPoint31FromLatLon(actual);
		PointI wanted31 = NativeUtilities.getPoint31FromLatLon(center);
		int dx = wanted31.getX() - actual31.getX();
		int dy = wanted31.getY() - actual31.getY();
		// Nudge the target by the observed offset (works in any projection:
		// the mapping is locally linear at these sub-pixel magnitudes).
		PointI current = NativeUtilities.getPoint31FromLatLon(center);
		offscreenView.setMapTargetLocation(new PointI(current.getX() + dx, current.getY() + dy));
		Log.i(TAG, "recenter: actual=" + actual + " drift31=(" + dx + "," + dy + ")");
		return distanceMeters(actual, center);
	}

	private static double distanceMeters(@NonNull LatLon from, @NonNull LatLon to) {
		double metersPerDegLat = 111320.0;
		double metersPerDegLon = 111320.0 * Math.cos(Math.toRadians(to.getLatitude()));
		return Math.hypot((from.getLatitude() - to.getLatitude()) * metersPerDegLat,
				(from.getLongitude() - to.getLongitude()) * metersPerDegLon);
	}

	private void waitForStable(long deadline) {
		int lastFrameId = -1;
		int lastReported = -1;
		long stableSince = 0;
		String lastReason = null;
		while (SystemClock.uptimeMillis() < deadline) {
			int frameId = offscreenView != null ? offscreenView.getFrameId() : -1;
			// The renderer exposes no tile-count total; the monotonic frame
			// id (advancing as loaded data arrives) is the progress signal.
			if (frameId != lastReported) {
				reportProgress(R.string.topoprint_progress_tiles_frames, Math.max(frameId, 0));
				lastReported = frameId;
			}
			// Activity is anything that says work is still pending: a new
			// frame, or the renderer reporting it is not idle (data arriving,
			// symbols re-layout, elevation … — MapRendererView.isIdle(),
			// available since the 5.4 core; getNotIdleReason() tells why).
			// Frame-id quietness alone mistook inter-batch loading gaps for
			// a settled state on slow devices, dropping the label/icon
			// batches that load last (top/left of the raster).
			long now = SystemClock.uptimeMillis();
			boolean idle = offscreenView == null || offscreenView.isIdle();
			boolean active = !idle || frameId != lastFrameId;
			if (active) {
				if (!idle) {
					String reason = offscreenView.getNotIdleReason();
					if (reason != null && !reason.equals(lastReason)) {
						Log.i(TAG, "not idle: " + reason);
						lastReason = reason;
					}
				}
				lastFrameId = frameId;
				stableSince = now;
			} else if (now - stableSince >= FRAME_STABLE_MS && now - stableSince > 0
					&& now - startUptimeMs >= MIN_SOAK_MS) {
				break;
			}
			SystemClock.sleep(FRAME_POLL_MS);
		}
	}

	/**
	 * Measures actualGroundWidth / requestedGroundWidth by raycasting the
	 * raster corners through the renderer. Runs on the UI thread.
	 */
	@Nullable
	private Double measureCoverageRatioOnUi() {
		final CountDownLatch latch = new CountDownLatch(1);
		final Double[] ratioHolder = new Double[]{null};
		mainHandler.post(() -> {
			ratioHolder[0] = measureCoverageRatio();
			latch.countDown();
		});
		try {
			latch.await(3, TimeUnit.SECONDS);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
		return ratioHolder[0];
	}

	private void adjustZoomOnUiThread(double deltaZoom) {
		final CountDownLatch latch = new CountDownLatch(1);
		mainHandler.post(() -> {
			// Base on the last requested fractional zoom, not getZoom():
			// the core's getter includes a surface/terrain offset that
			// drifts as elevation tiles load, which poisoned the
			// calibration's delta and froze it at a wrong coverage.
			if (offscreenView != null) {
				applyFractionalZoom(lastTargetFractionalZoom + deltaZoom);
			}
			latch.countDown();
		});
		try {
			latch.await(2, TimeUnit.SECONDS);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
	}

	/**
	 * Applies a fractional zoom, decomposed into {@code zoomLevel +
	 * visualZoom} with round-to-nearest — identical to the EyePiece CLI and
	 * the plain float setZoom path. Round (not ceil) keeps the fraction in
	 * ±0.5, so visualZoom stays near 1.0 where the core's zoom resolution
	 * and its min-zoom clamps behave; a ceil decomposition pushed visualZoom
	 * against its 0.5 lower bound on 1:10000 maps and the calibration
	 * could no longer correct the coverage. The ground coverage stays exact
	 * because zoomLevel + visualZoom-fraction equals the requested
	 * fractional zoom.
	 */
	private void applyFractionalZoom(double fractionalZoom) {
		int zoomLevel = (int) Math.round(fractionalZoom);
		float fraction = (float) (fractionalZoom - zoomLevel); // [-0.5, +0.5]
		// Inverse of MapRendererView.getZoom(): fraction >= 0 maps to
		// visualZoom in [1, 2], fraction < 0 to visualZoom in [0.5, 1).
		float visualZoom = fraction >= 0f ? 1f + fraction : 1f + fraction / 2f;
		lastTargetFractionalZoom = fractionalZoom;
		if (offscreenView != null) {
			offscreenView.setZoom(ZoomLevel.swigToEnum(zoomLevel), visualZoom);
		}
		Log.i(TAG, "setZoom: fractional=" + fractionalZoom + " zoomLevel=" + zoomLevel
				+ " visualZoom=" + visualZoom);
	}

	/** Runs on the UI thread; null when the raycast failed. */
	@Nullable
	private Double measureCoverageRatio() {
		if (offscreenView == null) {
			return null;
		}
		int w = rasterWidthPx();
		int h = rasterHeightPx();
		LatLon tl = NativeUtilities.getLatLonFromPixel(offscreenView, null, 0, 0);
		LatLon tr = NativeUtilities.getLatLonFromPixel(offscreenView, null, w, 0);
		LatLon bl = NativeUtilities.getLatLonFromPixel(offscreenView, null, 0, h);
		LatLon br = NativeUtilities.getLatLonFromPixel(offscreenView, null, w, h);
		if (tl == null || tr == null || bl == null || br == null) {
			return null;
		}
		int zone = UtmMath.INSTANCE.fromLatLon(center.getLatitude(), center.getLongitude()).getZoneNumber();
		double eastTopLeft = UtmMath.INSTANCE.fromLatLon(tl.getLatitude(), tl.getLongitude(), zone).getEasting();
		double eastTopRight = UtmMath.INSTANCE.fromLatLon(tr.getLatitude(), tr.getLongitude(), zone).getEasting();
		double eastBottomLeft = UtmMath.INSTANCE.fromLatLon(bl.getLatitude(), bl.getLongitude(), zone).getEasting();
		double eastBottomRight = UtmMath.INSTANCE.fromLatLon(br.getLatitude(), br.getLongitude(), zone).getEasting();
		double northTopLeft = UtmMath.INSTANCE.fromLatLon(tl.getLatitude(), tl.getLongitude(), zone).getNorthing();
		double northTopRight = UtmMath.INSTANCE.fromLatLon(tr.getLatitude(), tr.getLongitude(), zone).getNorthing();
		double northBottomLeft = UtmMath.INSTANCE.fromLatLon(bl.getLatitude(), bl.getLongitude(), zone).getNorthing();
		double northBottomRight = UtmMath.INSTANCE.fromLatLon(br.getLatitude(), br.getLongitude(), zone).getNorthing();
		double measuredWidth = ((eastTopRight + eastBottomRight) - (eastTopLeft + eastBottomLeft)) / 2.0;
		double measuredHeight = ((northTopLeft + northTopRight) - (northBottomLeft + northBottomRight)) / 2.0;
		Log.i(TAG, "coverage measure: measured=" + measuredWidth + "x" + measuredHeight
				+ " requested=" + widthMeters + "x" + heightMeters);
		if (measuredWidth <= 0 || measuredHeight <= 0) {
			return null;
		}
		// Blend both axes (the zoom scales them together in the renderer).
		return (measuredWidth / widthMeters + measuredHeight / heightMeters) / 2.0;
	}

	@Override
	public void onUpdateFrame(@NonNull MapRendererView mapRendererView) {
	}

	@Override
	public void onFrameReady(@NonNull MapRendererView mapRendererView) {
		frameReady = true;
	}

	/**
	 * Runs on the UI thread. The renderer has been observed to alternate
	 * between two camera states — the corrected one and one ~1.5 km off — on
	 * consecutive raycasts, so a single sample is a coin flip. Sample the
	 * centre several times (the cleanest sample decides), then verify one
	 * corner burst (all five raycasts of a burst land within the same camera
	 * state) before grabbing the bitmap in the very same window.
	 *
	 * Fail-loud rules instead of the old silent fallbacks:
	 * - a drift beyond the overscan margin means the raster cannot cover the
	 *   map frame — delivering it would print blank bands;
	 * - missing raycast corners mean the georeference would be an assumption,
	 *   not a measurement — a mis-stated georef silently shifts every grid
	 *   line against the map features.
	 */
	private void captureNow(int attempt) {
		if (delivered || offscreenView == null) return;
		reportProgress(R.string.topoprint_progress_wait, attempt + 1, CAPTURE_RETRIES + 1);
		int w = rasterWidthPx();
		int h = rasterHeightPx();
		double bestDrift = Double.MAX_VALUE;
		for (int i = 0; i < DRIFT_SAMPLES; i++) {
			LatLon sample = NativeUtilities.getLatLonFromPixel(offscreenView, null, w / 2, h / 2);
			if (sample != null) {
				bestDrift = Math.min(bestDrift, distanceMeters(sample, center));
			}
			if (bestDrift <= centerToleranceMeters()) {
				break;
			}
			SystemClock.sleep(DRIFT_SAMPLE_INTERVAL_MS);
		}
		Log.i(TAG, "capture gate " + attempt + ": bestDrift=" + bestDrift + " m (" + DRIFT_SAMPLES + " samples)");
		if (bestDrift > centerToleranceMeters() && attempt < CAPTURE_RETRIES) {
			if (bestDrift != Double.MAX_VALUE) {
				// Nudge the target by the residual offset and let a fresh
				// frame settle before the next gate.
				LatLon actual = NativeUtilities.getLatLonFromPixel(offscreenView, null, w / 2, h / 2);
				if (actual != null) {
					PointI actual31 = NativeUtilities.getPoint31FromLatLon(actual);
					PointI wanted31 = NativeUtilities.getPoint31FromLatLon(center);
					PointI current31 = NativeUtilities.getPoint31FromLatLon(center);
					offscreenView.setMapTargetLocation(new PointI(
							current31.getX() + wanted31.getX() - actual31.getX(),
							current31.getY() + wanted31.getY() - actual31.getY()));
				}
			}
			mainHandler.postDelayed(() -> captureNow(attempt + 1), CAPTURE_RETRY_DELAY_MS);
			return;
		}
		// Verified window: a self-consistent burst (centre before and after the
		// corners, and the corner average agreeing with both) proves the five
		// raycasts saw one camera state; then the bitmap is grabbed inside the
		// very same window.
		for (int burst = 0; burst < CAPTURE_BURSTS; burst++) {
			LatLon ctr1 = NativeUtilities.getLatLonFromPixel(offscreenView, null, w / 2, h / 2);
			LatLon tl = NativeUtilities.getLatLonFromPixel(offscreenView, null, 0, 0);
			LatLon tr = NativeUtilities.getLatLonFromPixel(offscreenView, null, w, 0);
			LatLon bl = NativeUtilities.getLatLonFromPixel(offscreenView, null, 0, h);
			LatLon br = NativeUtilities.getLatLonFromPixel(offscreenView, null, w, h);
			LatLon ctr2 = NativeUtilities.getLatLonFromPixel(offscreenView, null, w / 2, h / 2);
			if (ctr1 == null || ctr2 == null || tl == null || tr == null || bl == null || br == null) {
				Log.w(TAG, "burst " + burst + ": raycast failed — refusing to guess the georeference");
				continue;
			}
			LatLon cornerAvg = new LatLon(
					(tl.getLatitude() + tr.getLatitude() + bl.getLatitude() + br.getLatitude()) / 4.0,
					(tl.getLongitude() + tr.getLongitude() + bl.getLongitude() + br.getLongitude()) / 4.0);
			// The Mercator viewport's corner average sits slightly off the UTM
			// rectangle centre; the offset grows with coverage (~2 m at
			// 1:25000 portrait, ~2.1 m at 1:50000 landscape — well under
			// 0.05 mm on paper). Scale the agreement window with the coverage
			// so the check still catches camera-state flips (~1.5 km apart)
			// without rejecting large maps for a systematic offset.
			double cornerAgreementMeters = Math.max(2.0,
					0.002 * Math.max(widthMeters, heightMeters));
			boolean stable = distanceMeters(ctr1, ctr2) <= 2.0
					&& distanceMeters(ctr1, cornerAvg) <= cornerAgreementMeters
					&& distanceMeters(ctr1, center) <= centerToleranceMeters();
			boolean covered = rasterCoversMapFrame(tl, tr, br, bl);
			Log.i(TAG, "burst " + burst + ": stable=" + stable + " covered=" + covered
					+ " drift=" + distanceMeters(ctr1, center) + " m center=" + ctr1);
			if (stable && covered) {
				Bitmap bitmap = offscreenView.getBitmap();
				if (bitmap == null) {
					Log.w(TAG, "getBitmap returned null in a verified window");
					continue;
				}
				// Post-capture same-state gate: re-raycast the centre right
				// after the bitmap was grabbed. If the camera state flipped
				// between the corner raycasts and the frame grab, the bitmap
				// content would not match the georef corners above — deliver
				// nothing in that case, retry in a fresh window instead.
				LatLon ctr3 = NativeUtilities.getLatLonFromPixel(offscreenView, null, w / 2, h / 2);
				double sameStateDeltaM = ctr3 == null ? -1.0 : distanceMeters(ctr1, ctr3);
				if (ctr3 != null && sameStateDeltaM > centerToleranceMeters()) {
					Log.w(TAG, String.format("burst %d: camera state flipped after the frame grab " +
							"(postCaptureDeltaM=%.1f) — discarding the bitmap", burst, sameStateDeltaM));
					bitmap.recycle();
					continue;
				}
				OffscreenMapRenderer.MapGeoref georef =
						new OffscreenMapRenderer.MapGeoref(tl, tr, br, bl, w, h);
				Log.i(TAG, "captured in verified window: bitmapPx=" + w + "x" + h
						+ " zoom=" + offscreenView.getZoom()
						+ " corners: tl=" + tl + " tr=" + tr + " br=" + br + " bl=" + bl
						+ String.format(" postCaptureDeltaM=%.1f", sameStateDeltaM));
				restore();
				reportProgress(R.string.topoprint_progress_capture);
				listener.onRendered(new OffscreenMapRenderer.RenderResult(bitmap, georef));
				return;
			}
		}
		if (attempt < CAPTURE_RETRIES) {
			mainHandler.postDelayed(() -> captureNow(attempt + 1), CAPTURE_RETRY_DELAY_MS);
			return;
		}
		Log.w(TAG, "no verified capture window after " + (CAPTURE_RETRIES + 1)
				+ " attempts — failing instead of delivering a mis-georeferenced map");
		restore();
		listener.onRendered(null);
	}

	/**
	 * The centre may wander as long as the raster rectangle (georeferenced by
	 * its true raycast corners) still contains the printed map frame
	 * ({@code params} map area centred at the requested centre); beyond that
	 * the map would show blank bands inside the frame.
	 */
	private double maxTolerableDriftMeters() {
		double marginX = (widthMeters - params.getMapWidthMeters()) / 2.0;
		double marginY = (heightMeters - params.getMapHeightMeters()) / 2.0;
		return Math.max(1.0, Math.min(marginX, marginY));
	}

	/** UTM containment check of the map frame inside the raster rectangle. */
	private boolean rasterCoversMapFrame(@NonNull LatLon tl, @NonNull LatLon tr,
	                                     @NonNull LatLon br, @NonNull LatLon bl) {
		int zone = UtmMath.INSTANCE.fromLatLon(center.getLatitude(), center.getLongitude()).getZoneNumber();
		double mapHalfW = params.getMapWidthMeters() / 2.0;
		double mapHalfH = params.getMapHeightMeters() / 2.0;
		double cE = UtmMath.INSTANCE.fromLatLon(center.getLatitude(), center.getLongitude(), zone).getEasting();
		double cN = UtmMath.INSTANCE.fromLatLon(center.getLatitude(), center.getLongitude(), zone).getNorthing();
		double eMin = cE - mapHalfW;
		double eMax = cE + mapHalfW;
		double nMin = cN - mapHalfH;
		double nMax = cN + mapHalfH;
		double[] eastings = {
				UtmMath.INSTANCE.fromLatLon(tl.getLatitude(), tl.getLongitude(), zone).getEasting(),
				UtmMath.INSTANCE.fromLatLon(tr.getLatitude(), tr.getLongitude(), zone).getEasting(),
				UtmMath.INSTANCE.fromLatLon(br.getLatitude(), br.getLongitude(), zone).getEasting(),
				UtmMath.INSTANCE.fromLatLon(bl.getLatitude(), bl.getLongitude(), zone).getEasting()};
		double[] northings = {
				UtmMath.INSTANCE.fromLatLon(tl.getLatitude(), tl.getLongitude(), zone).getNorthing(),
				UtmMath.INSTANCE.fromLatLon(tr.getLatitude(), tr.getLongitude(), zone).getNorthing(),
				UtmMath.INSTANCE.fromLatLon(br.getLatitude(), br.getLongitude(), zone).getNorthing(),
				UtmMath.INSTANCE.fromLatLon(bl.getLatitude(), bl.getLongitude(), zone).getNorthing()};
		double rEMin = Double.MAX_VALUE, rEMax = -Double.MAX_VALUE;
		double rNMin = Double.MAX_VALUE, rNMax = -Double.MAX_VALUE;
		for (double e : eastings) {
			rEMin = Math.min(rEMin, e);
			rEMax = Math.max(rEMax, e);
		}
		for (double n : northings) {
			rNMin = Math.min(rNMin, n);
			rNMax = Math.max(rNMax, n);
		}
		// Mercator-north-up raster corners form a slightly rotated quadrilateral
		// in UTM; the axis-aligned raster rect bounds approximate the coverage.
		boolean covered = rEMin <= eMin && rEMax >= eMax && rNMin <= nMin && rNMax >= nMax;
		if (!covered) {
			Log.w(TAG, "frame rect in UTM: E[" + eMin + "," + eMax + "] N[" + nMin + "," + nMax
					+ "] raster rect: E[" + rEMin + "," + rEMax + "] N[" + rNMin + "," + rNMax + "]");
		}
		return covered;
	}

	private int rasterWidthPx() {
		return (int) Math.ceil(widthMeters / OffscreenMapRenderer.groundMetersPerPixel(params));
	}

	private int rasterHeightPx() {
		return (int) Math.ceil(heightMeters / OffscreenMapRenderer.groundMetersPerPixel(params));
	}

	private void finish(@Nullable Bitmap bitmap, int widthPx, int heightPx) {
		if (delivered) return;
		delivered = true;
		Log.w(TAG, "finish(): bitmap=" + (bitmap != null));
		mainHandler.removeCallbacksAndMessages(null);
		restore();
		listener.onRendered(bitmap == null ? null
				: new OffscreenMapRenderer.RenderResult(bitmap, null));
	}

	private void restore() {
		if (offscreenView != null) {
			if (listenerAttached) {
				// MapRendererView.removeListener() locks on its _mapRenderer
				// field, which is null when setupRenderer never completed —
				// only detach after a successful setup (the flag's write and
				// read both happen on the main thread).
				try {
					offscreenView.removeListener(this);
				} catch (RuntimeException e) {
					Log.w(TAG, "removeListener failed on teardown", e);
				}
				listenerAttached = false;
			}
			if (printContext != null) {
				printContext.setMapRendererView(null);
			}
			if (mainRendererView != null) {
				// Hand the EGL renderer back to the main view exactly like the
				// Android Auto → phone transition does. The handed-over EGL
				// thread now belongs to the main view again.
				mainRendererView.setupRenderer(app, 0, 0, offscreenView);
			}
			// Re-attach the main map BEFORE tearing the offscreen view down:
			// setMapRenderer() unconditionally strips the previous view's
			// providers through its internal renderer, so the offscreen view
			// must still be functional here.
			reattachMainMap();
			// Reclaim this view's own heavy state before dropping it. Without
			// this, repeated prints pile up per-print W×H×4 direct buffers
			// until a later print OOMs inside setupRenderer()'s
			// allocateDirect (the buffer is charged to the heap growth
			// limit), and the render callbacks that setupRenderer() pinned
			// natively keep the whole view reachable.
			// - disableFrameReader() drops the frame read-back buffer
			//   (_byteBuffer) immediately, so its memory frees at the next
			//   collection instead of living as long as the view;
			// - stopRenderer() cannot touch the main view's renderer: after
			//   the hand-over above this view is flagged isHandedOver, so the
			//   call only clears the stale _mapRenderer/EGL-thread references
			//   and unpins this view's RenderRequestCallback and GPU-worker
			//   prologue/epilogue from the (surviving) native renderer. On
			//   the failed-setup path (no renderer at all) it degrades to the
			//   same unpins and a logged warning.
			offscreenView.disableFrameReader();
			offscreenView.stopRenderer();
			offscreenView = null;
		} else {
			// The render failed before the offscreen view existed (e.g. the
			// raster size was refused in step1) — the main map is still
			// detached from step1's freeze and must be re-attached here.
			reattachMainMap();
		}
		printContext = null;
		if (savedContext != null) {
			// Recreate the global context with the saved density and bind it to
			// the main view. Note: use the NEW instance — setMapRendererView on
			// the old one would early-return (it still refers to the main view)
			// and the raster providers would never come back.
			NativeCoreContext.setMapRendererContext(app, savedDensity);
			MapRendererContext restoredContext = NativeCoreContext.getMapRendererContext();
			if (restoredContext != null) {
				// When the supersampling matches the device density, the print
				// ran on the live map's own context instance — drop the print
				// presentation overrides and rebuild, so the live map resumes
				// with the user's settings (a fresh instance is unaffected:
				// clearing is a no-op and the rebuild keeps its presentation).
				restoredContext.setMapDensityOverride(null);
				restoredContext.setSymbolsScaleOverride(null);
				restoredContext.updateMapSettings(false);
				if (mainRendererView != null) {
					restoredContext.setMapRendererView(mainRendererView);
				}
			}
		}
		tileView = null;
	}

	/**
	 * Re-attaches the main map to the main renderer view; the layers migrate
	 * their providers back. Nulls {@code tileView} afterwards, so at most one
	 * call per restore has an effect (the offscreen branch and the no-view
	 * branch are mutually exclusive, this guards against future reordering).
	 */
	private void reattachMainMap() {
		if (tileView != null && mainRendererView != null) {
			tileView.setMapRenderer(mainRendererView, false);
			tileView = null;
		}
	}
}
