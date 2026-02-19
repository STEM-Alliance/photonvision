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

package org.photonvision.vision.pipe.impl;

import edu.wpi.first.apriltag.AprilTagDetection;
import java.util.List;
import org.opencv.core.Mat;
import org.photonvision.jni.GpuDetectorJNI;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.opencv.Releasable;
import org.photonvision.vision.pipe.CVPipe;

public class GpuAprilTagDetectionPipe extends CVPipe<CVMat, List<AprilTagDetection>, Void>
        implements Releasable {

    private long handle = -1;
    private int currentWidth = -1;
    private int currentHeight = -1;

    private double fx, cx, fy, cy;
    private double k1, k2, p1, p2, k3;
    private boolean paramsDirty = false;

    public GpuAprilTagDetectionPipe() {}

    @Override
    protected List<AprilTagDetection> process(CVMat in) {
        Mat mat = in.getMat();
        if (mat.empty()) {
            return List.of();
        }

        int width = mat.cols();
        int height = mat.rows();

        if (handle == -1 || width != currentWidth || height != currentHeight) {
            if (handle != -1) {
                GpuDetectorJNI.destroyGpuDetector(handle);
            }
            handle = GpuDetectorJNI.createGpuDetector(width, height);
            currentWidth = width;
            currentHeight = height;
            // Need to re-set params on new detector
            paramsDirty = true;
        }

        if (paramsDirty) {
            GpuDetectorJNI.setparams(handle, fx, cx, fy, cy, k1, k2, p1, p2, k3);
            paramsDirty = false;
        }

        AprilTagDetection[] detectionsLine =
                GpuDetectorJNI.processimage(handle, mat.getNativeObjAddr());

        if (detectionsLine == null) {
            return List.of();
        }

        return List.of(detectionsLine);
    }

    public void setCameraParams(
            double fx,
            double cx,
            double fy,
            double cy,
            double k1,
            double k2,
            double p1,
            double p2,
            double k3) {

        if (this.fx != fx
                || this.cx != cx
                || this.fy != fy
                || this.cy != cy
                || this.k1 != k1
                || this.k2 != k2
                || this.p1 != p1
                || this.p2 != p2
                || this.k3 != k3) {

            this.fx = fx;
            this.cx = cx;
            this.fy = fy;
            this.cy = cy;
            this.k1 = k1;
            this.k2 = k2;
            this.p1 = p1;
            this.p2 = p2;
            this.k3 = k3;
            paramsDirty = true;
        }
    }

    @Override
    public void release() {
        if (handle != -1) {
            GpuDetectorJNI.destroyGpuDetector(handle);
            handle = -1;
        }
    }
}
