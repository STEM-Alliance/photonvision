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

package org.photonvision.common.dataflow.networktables;

import edu.wpi.first.networktables.BooleanPublisher;
import edu.wpi.first.networktables.BooleanSubscriber;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableEvent.Kind;
import edu.wpi.first.networktables.NetworkTableInstance;
import java.util.EnumSet;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.vision.processes.VisionSourceManager;

/**
 * Manages global video recording state via NetworkTables. Recording topics are published at the
 * root /photonvision table level so that a single command starts/stops recording on all cameras
 * simultaneously.
 *
 * <p>NT topics:
 *
 * <ul>
 *   <li>{@code /photonvision/recordingRequest} - robot code sets this to true/false
 *   <li>{@code /photonvision/recordingState} - reflects actual recording state
 * </ul>
 */
public class NTVideoRecordingManager {
    private static final Logger logger =
            new Logger(NTVideoRecordingManager.class, LogGroup.NetworkTables);

    private final BooleanPublisher recordingStatePublisher;
    private final BooleanSubscriber recordingRequestSubscriber;

    private boolean currentRecordingState = false;

    public NTVideoRecordingManager(NetworkTable rootTable, NetworkTableInstance ntInstance) {
        recordingStatePublisher = rootTable.getBooleanTopic("recordingState").publish();
        recordingRequestSubscriber = rootTable.getBooleanTopic("recordingRequest").subscribe(false);

        // Publish default so the topic shows up in NT
        recordingRequestSubscriber.getTopic().publish().setDefault(false);
        recordingStatePublisher.set(false);

        // Add listener for recording request changes
        ntInstance.addListener(
                recordingRequestSubscriber,
                EnumSet.of(Kind.kValueAll),
                event -> {
                    boolean requested = event.valueData.value.getBoolean();
                    onRecordingRequestChanged(requested);
                });

        logger.info("NTVideoRecordingManager initialized");
    }

    private void onRecordingRequestChanged(boolean requested) {
        if (requested == currentRecordingState) return;

        logger.info("Recording request changed to: " + requested);

        var modules = VisionSourceManager.getInstance().getVisionModules();
        if (modules.isEmpty()) {
            logger.warn("No vision modules available for recording");
            return;
        }

        if (requested) {
            for (var module : modules) {
                module.startRecording();
            }
            currentRecordingState = true;
            logger.info("Recording started on " + modules.size() + " camera(s)");
        } else {
            for (var module : modules) {
                module.stopRecording();
            }
            currentRecordingState = false;
            logger.info("Recording stopped on " + modules.size() + " camera(s)");
        }

        recordingStatePublisher.set(currentRecordingState);
    }

    /** Publish the current recording state to NT. Called periodically. */
    public void updateRecordingState() {
        // Check if any module is still actually recording (could have stopped due to disk full)
        boolean anyRecording = false;
        for (var module : VisionSourceManager.getInstance().getVisionModules()) {
            if (module.isRecording()) {
                anyRecording = true;
                break;
            }
        }

        if (anyRecording != currentRecordingState) {
            currentRecordingState = anyRecording;
            recordingStatePublisher.set(currentRecordingState);

            if (!anyRecording) {
                logger.info("All recordings stopped (possibly due to disk space)");
            }
        }
    }

    public void close() {
        // Stop any active recording before closing
        if (currentRecordingState) {
            for (var module : VisionSourceManager.getInstance().getVisionModules()) {
                module.stopRecording();
            }
            currentRecordingState = false;
        }
        recordingStatePublisher.close();
        recordingRequestSubscriber.close();
    }
}
