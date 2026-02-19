/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.vision.frame.consumer;

import edu.wpi.first.networktables.IntegerSubscriber;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.StringSubscriber;
import edu.wpi.first.wpilibj.DriverStation.MatchType;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import java.util.function.Supplier;
import org.opencv.core.Mat;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.dataflow.networktables.NetworkTablesManager;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.vision.opencv.CVMat;

/**
 * Records processed output frames to MKV video files using GStreamer with the Jetson's nvjpeg
 * hardware encoder. The GStreamer pipeline accepts raw BGR frames via stdin and encodes them with
 * nvjpegenc into a Matroska container.
 */
public class VideoRecordingConsumer implements Consumer<CVMat> {
    private final Logger logger = new Logger(VideoRecordingConsumer.class, LogGroup.General);

    private static final String FILE_PATH = ConfigManager.getInstance().getImageSavePath().toString();
    private static final String FILE_EXTENSION = ".mkv";
    private static final double MAX_DISK_USAGE_FRACTION = 0.90;
    private static final long DISK_CHECK_INTERVAL_MS = 5000;

    private static final DateFormat df = new SimpleDateFormat("yyyy-MM-dd");
    private static final DateFormat tf = new SimpleDateFormat("HHmmss");

    private final Supplier<String> cameraNicknameSupplier;
    private final String cameraUniqueName;

    private StringSubscriber ntEventName;
    private IntegerSubscriber ntMatchNum;
    private IntegerSubscriber ntMatchType;

    private Process gstreamerProcess;
    private OutputStream gstreamerStdin;
    private final AtomicBoolean recording = new AtomicBoolean(false);

    private int frameWidth = -1;
    private int frameHeight = -1;
    private long lastDiskCheckTime = 0;
    private String currentRecordingPath = "";

    // Pre-allocated buffer for frame data to avoid per-frame allocation
    private byte[] frameBuffer;

    public VideoRecordingConsumer(Supplier<String> cameraNicknameSupplier, String cameraUniqueName) {
        this.cameraNicknameSupplier = cameraNicknameSupplier;
        this.cameraUniqueName = cameraUniqueName;

        NetworkTable fmsTable = NetworkTablesManager.getInstance().getNTInst().getTable("FMSInfo");
        this.ntEventName = fmsTable.getStringTopic("EventName").subscribe("UNKNOWN");
        this.ntMatchNum = fmsTable.getIntegerTopic("MatchNumber").subscribe(0);
        this.ntMatchType = fmsTable.getIntegerTopic("MatchType").subscribe(0);
    }

    @Override
    public void accept(CVMat image) {
        if (!recording.get()) return;

        if (image == null || image.getMat() == null || image.getMat().empty()) return;

        Mat mat = image.getMat();

        // Initialize on first frame if process not started yet
        if (gstreamerProcess == null || !gstreamerProcess.isAlive()) {
            if (!startGstreamerProcess(mat.cols(), mat.rows())) {
                return;
            }
        }

        // Periodic disk space check
        long now = System.currentTimeMillis();
        if (now - lastDiskCheckTime > DISK_CHECK_INTERVAL_MS) {
            lastDiskCheckTime = now;
            if (isDiskFull()) {
                logger.warn(
                        "Disk usage exceeds 90%, stopping recording for " + cameraNicknameSupplier.get());
                stopRecording();
                return;
            }
        }

        try {
            // Get raw BGR bytes from the Mat
            int dataSize = (int) (mat.total() * mat.elemSize());
            if (frameBuffer == null || frameBuffer.length != dataSize) {
                frameBuffer = new byte[dataSize];
            }
            mat.get(0, 0, frameBuffer);

            gstreamerStdin.write(frameBuffer);
        } catch (IOException e) {
            logger.error(
                    "Error writing frame to GStreamer process for " + cameraNicknameSupplier.get(), e);
            stopRecording();
        }
    }

    /**
     * Start recording video. The actual GStreamer process is spawned lazily when the first frame
     * arrives, since we need to know the frame dimensions.
     */
    public void startRecording() {
        if (recording.getAndSet(true)) {
            logger.info("Recording already active for " + cameraNicknameSupplier.get());
            return;
        }

        if (isDiskFull()) {
            logger.warn(
                    "Cannot start recording for "
                            + cameraNicknameSupplier.get()
                            + ": disk usage exceeds 90%");
            recording.set(false);
            return;
        }

        // Reset dimensions so they're detected from the next frame
        frameWidth = -1;
        frameHeight = -1;
        lastDiskCheckTime = System.currentTimeMillis();
        logger.info(
                "Recording armed for " + cameraNicknameSupplier.get() + ", will start on first frame");
    }

    /** Stop the recording and finalize the MKV file. */
    public void stopRecording() {
        if (!recording.getAndSet(false)) {
            return;
        }

        destroyGstreamerProcess();
        logger.info("Recording stopped for " + cameraNicknameSupplier.get());
        if (!currentRecordingPath.isEmpty()) {
            logger.info("Video saved: " + currentRecordingPath);
            currentRecordingPath = "";
        }
    }

    /** Returns whether this consumer is currently recording. */
    public boolean isRecording() {
        return recording.get();
    }

    /**
     * Spawn the GStreamer process with the detected frame dimensions.
     *
     * @return true if the process was started successfully
     */
    private boolean startGstreamerProcess(int width, int height) {
        this.frameWidth = width;
        this.frameHeight = height;

        // Build output file path
        var now = new Date();
        String matchData = getMatchData();
        String fileName =
                cameraNicknameSupplier.get()
                        + "_"
                        + df.format(now)
                        + "T"
                        + tf.format(now)
                        + "_"
                        + matchData;

        File cameraDir = new File(FILE_PATH, this.cameraUniqueName);
        if (!cameraDir.exists()) {
            cameraDir.mkdirs();
        }
        currentRecordingPath = cameraDir.toPath().resolve(fileName + FILE_EXTENSION).toString();

        // Build GStreamer pipeline command
        // nvjpegenc accepts I420/NV12/YV12/GRAY8, so we need videoconvert from BGR
        int fps = 30; // Default FPS

        List<String> cmd = new ArrayList<>();
        cmd.add("gst-launch-1.0");
        cmd.add("-e");
        cmd.add("fdsrc");
        cmd.add("!");
        cmd.add("rawvideoparse");
        cmd.add("width=" + width);
        cmd.add("height=" + height);
        cmd.add("format=bgr");
        cmd.add("framerate=" + fps + "/1");
        cmd.add("!");
        cmd.add("videoconvert");
        cmd.add("!");
        cmd.add("nvjpegenc");
        cmd.add("quality=85");
        cmd.add("!");
        cmd.add("matroskamux");
        cmd.add("!");
        cmd.add("filesink");
        // Passing location as a single argument avoids shell interpretation of special chars
        cmd.add("location=" + currentRecordingPath);

        logger.info("Starting GStreamer recording: " + String.join(" ", cmd));

        try {
            ProcessBuilder pb = new ProcessBuilder(cmd).redirectErrorStream(true);
            gstreamerProcess = pb.start();
            gstreamerStdin = gstreamerProcess.getOutputStream();

            // Start a thread to drain stdout/stderr and log it
            Thread drainThread =
                    new Thread(
                            () -> {
                                try (java.io.BufferedReader reader =
                                        new java.io.BufferedReader(
                                                new java.io.InputStreamReader(gstreamerProcess.getInputStream()))) {
                                    String line;
                                    while ((line = reader.readLine()) != null) {
                                        logger.info("GST: " + line);
                                    }
                                } catch (IOException e) {
                                    // Process ending
                                }
                            },
                            "GStreamer-drain-" + cameraNicknameSupplier.get());
            drainThread.setDaemon(true);
            drainThread.start();

            logger.info(
                    "GStreamer recording started for "
                            + cameraNicknameSupplier.get()
                            + " ("
                            + width
                            + "x"
                            + height
                            + " @ "
                            + fps
                            + "fps) -> "
                            + currentRecordingPath);
            return true;
        } catch (IOException e) {
            logger.error("Failed to start GStreamer process for " + cameraNicknameSupplier.get(), e);
            recording.set(false);
            return false;
        }
    }

    /** Gracefully shut down the GStreamer process. */
    private void destroyGstreamerProcess() {
        if (gstreamerStdin != null) {
            try {
                gstreamerStdin.close();
            } catch (IOException e) {
                logger.warn(
                        "Error closing GStreamer stdin for "
                                + cameraNicknameSupplier.get()
                                + ": "
                                + e.getMessage());
            }
            gstreamerStdin = null;
        }

        if (gstreamerProcess != null) {
            try {
                // Give GStreamer time to finalize the MKV after stdin closes
                boolean exited = gstreamerProcess.waitFor(5, java.util.concurrent.TimeUnit.SECONDS);
                if (!exited) {
                    logger.warn(
                            "GStreamer process did not exit cleanly for "
                                    + cameraNicknameSupplier.get()
                                    + ", force killing");
                    gstreamerProcess.destroyForcibly();
                }
            } catch (InterruptedException e) {
                gstreamerProcess.destroyForcibly();
                Thread.currentThread().interrupt();
            }
            gstreamerProcess = null;
        }

        frameBuffer = null;
    }

    /**
     * Returns the match data collected from NT. E.g., "Q58-CASJ" for qualification match 58 at the
     * CASJ event.
     */
    private String getMatchData() {
        var matchType = ntMatchType.getAtomic();
        var matchNum = ntMatchNum.getAtomic();
        var eventName = ntEventName.getAtomic();

        MatchType wpiMatchType = MatchType.None;
        if (matchType.value >= 0 && matchType.value < MatchType.values().length) {
            wpiMatchType = MatchType.values()[(int) matchType.value];
        }

        return wpiMatchType.name() + "-" + matchNum.value + "-" + eventName.value;
    }

    /** Check if the disk is at or above the 90% usage threshold. */
    private boolean isDiskFull() {
        File root = new File(FILE_PATH);
        long totalSpace = root.getTotalSpace();
        long usableSpace = root.getUsableSpace();
        if (totalSpace == 0) return false;
        double usedFraction = 1.0 - ((double) usableSpace / totalSpace);
        return usedFraction >= MAX_DISK_USAGE_FRACTION;
    }

    /** Release all resources. */
    public void close() {
        stopRecording();
        if (ntEventName != null) ntEventName.close();
        if (ntMatchNum != null) ntMatchNum.close();
        if (ntMatchType != null) ntMatchType.close();
    }
}
