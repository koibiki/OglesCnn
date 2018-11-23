package com.example.eglnn.layer;

import android.content.Context;

import com.example.eglnn.Render;
import com.example.eglnn.eglenv.GLES31BackEnv;
import com.example.eglnn.utils.DataUtils;
import com.example.eglnn.utils.ShaderUtils;
import com.example.eglnn.utils.TestDataCreator;
import com.example.eglnn.utils.Utils;

import java.nio.FloatBuffer;
import java.util.Locale;

import static com.example.eglnn.Render.initCompPro;
import static com.example.eglnn.utils.Constants.S_DEPTH_WISE_CONV_SHADER_HEADER;

/**
 * depth wise conv
 * kernel 存储通道排列为 in_channel * channel_multiplier
 * 输出通道存储排列为 channel_multiplier * in_channel
 * tensorflow 例子
 * img1 = tf.constant(value=[[[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]]]],dtype=tf.float32)
 * img2 = tf.constant(value=[[[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]]]],dtype=tf.float32)
 * img = tf.concat(values=[img1,img2],axis=3)
 * filter1 = tf.constant(value=0, shape=[3,3,1,1],dtype=tf.float32)
 * filter2 = tf.constant(value=1, shape=[3,3,1,1],dtype=tf.float32)
 * filter3 = tf.constant(value=2, shape=[3,3,1,1],dtype=tf.float32)
 * filter4 = tf.constant(value=3, shape=[3,3,1,1],dtype=tf.float32)
 * filter_out1 = tf.concat(values=[filter1,filter2],axis=2)
 * filter_out2 = tf.concat(values=[filter3,filter4],axis=2)
 * filter_all = tf.concat(values=[filter_out1,filter_out2],axis=3)
 *
 * point_filter1 = tf.constant(value=[[[[0],[1],[2],[3]]]], shape=[1,1,4,1],dtype=tf.float32)
 * point_filter2 = tf.constant(value=1, shape=[1,1,4,1],dtype=tf.float32)
 * point_filter3 = tf.constant(value=2, shape=[1,1,4,1],dtype=tf.float32)
 * point_filter4 = tf.constant(value=3, shape=[1,1,4,1],dtype=tf.float32)
 * point_filter5 = tf.constant(value=4, shape=[1,1,4,1],dtype=tf.float32)
 * point_filter = tf.concat(values=[point_filter1,point_filter2,point_filter3,point_filter4,point_filter5],axis=3)
 * out_img_0 = tf.nn.depthwise_conv2d(input=img, filter=filter_all, strides=[1,1,1,1],rate=[1,1], padding='VALID')
 * out_img_1 = tf.nn.conv2d(input=out_img_0, filter=point_filter, strides=[1,1,1,1], padding='VALID')
 * out_img_2 = tf.nn.separable_conv2d(input=img, depthwise_filter=filter_all, pointwise_filter=point_filter,strides=[1,1,1,1], rate=[1,1], padding='VALID')
 */
public class DepthWiseConv extends Layer {

    private static final String TAG = "ConvGEMM";

    private int mKernelMultiplier;
    private FloatBuffer mOut;       // 每次只能读取一个深度上的数据
    private int mBindLayer;         // 纹理绑定深度编号

    private float[][][] mKernels;
    private int[] mStrides;
    private int[] mKernelShape;
    private int mShaderPro;
    private int mNumGroupsX;
    private int mNumGroupsY;
    private int mNumGroupsZ;
    private int[] mParams;
    private ActiveType mType;
    private PaddingType mPadType;

    private int mKernelTex;
    private int mPadW, mPadH;

    private DepthWiseConv(Context context, String name, Layer preLayer, int kernelMultipliter, int k_w, int k_h, int stride_w, int stride_h, ActiveType type) {
        super(context, name, preLayer);
        this.mKernelShape = new int[]{k_w, k_h, preLayer.getOutputShape()[2] * kernelMultipliter};
        this.mKernelMultiplier = kernelMultipliter;
        this.mStrides = new int[]{stride_w, stride_h};
        this.mType = type;
    }

    public DepthWiseConv(Context context, String name, Layer preLayer, int kernelMultipliter, int k_w, int k_h, PaddingType padType, int stride_w, int stride_h, ActiveType type) {
        this(context, name, preLayer, kernelMultipliter, k_w, k_h, stride_w, stride_h, type);
        this.mPadType = padType;
        this.mOutShape = calculateConvShapeByType(kernelMultipliter);
        this.mOut = FloatBuffer.allocate(mOutShape[0] * mOutShape[1] * 4);
        this.mBindLayer = 0;
//        this.mBindLayer = Utils.alignBy4(mOutShape[2]) / 4;
    }

    private int[] calculateConvShapeByType(int kernelMultipliter) {
        int width, height;
        if (mPadType == PaddingType.SAME) {
            width = (int) Math.ceil(mInShape[0] * 1.0f / mStrides[0]);
            mPadW = ((width - 1) * mStrides[0] + mKernelShape[0] - mInShape[0]) / 2;
            height = (int) Math.ceil(mInShape[1] * 1.0f / mStrides[1]);
            mPadH = ((height - 1) * mStrides[1] + mKernelShape[1] - mInShape[1]) / 2;
        } else {
            width = (int) Math.ceil((mInShape[0] - mKernelShape[0] + 1) * 1.0f / mStrides[0]);
            height = (int) Math.ceil((mInShape[1] - mKernelShape[1] + 1) * 1.0f / mStrides[1]);
        }
        return new int[]{width, height, mInShape[2] * kernelMultipliter};
    }

    private int getLocalSizeX(int[] outShape) {
        if (outShape[0] >= GLES31BackEnv.getMaxWorkGroupSize()) {
            return GLES31BackEnv.getMaxWorkGroupSize();
        } else {
            return outShape[0];
        }
    }

    private int getLocalSizeY(int[] outShape, int xSize) {
        int maxSize = GLES31BackEnv.getMaxWorkGroupSize() / xSize;
        if (outShape[1] <= maxSize) {
            return outShape[1];
        } else {
            return maxSize;
        }
    }

    private int getLocalSizeZ(int[] shape, int xSize, int ySize) {
        int maxZSize = GLES31BackEnv.getMaxWorkGroupSize() / (xSize * ySize) >= 64 ? 64 : GLES31BackEnv.getMaxWorkGroupSize() / (xSize * ySize);
        int zSize = Utils.alignBy4(shape[2]) / 4;
        if (zSize >= maxZSize) {
            return maxZSize;
        } else {
            return zSize;
        }
    }

    private void initConv() {
        int xSize = getLocalSizeX(mOutShape);
        mNumGroupsX = (int) Math.ceil(mOutShape[0] * 1.0f / xSize);

        int ySize = getLocalSizeY(mOutShape, xSize);
        mNumGroupsY = (int) Math.ceil(mOutShape[1] * 1.0f / ySize);

        int zSize = getLocalSizeZ(mInShape, xSize, ySize);
        mNumGroupsZ = (int) Math.ceil(Utils.alignBy4(mOutShape[2]) * 1.0f / 4 / zSize);

        String source = createShaderSource(xSize, ySize, zSize);
        mShaderPro = initCompPro(source);
        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createFloatTextureArray(mOutShape[0], mOutShape[1], Utils.alignBy4(mOutShape[2]) / 4);

        mKernels = createTestKernels();
//        mKernels = loadKennels();

        mKernelTex = Render.createFloatTextureArray(mKernelShape[0] * mKernelShape[1] + 1, 1, Utils.alignBy4(mKernelShape[2]) / 4);
        transferToKennelTex();
        createShaderParams();
    }

    private void transferToKennelTex() {
        for (int a = 0; a < 1; a++) {
            float[][] kennel = mKernels[a];
            for (int c = 0; c < kennel.length; c++) {
                Render.transferToTextureArrayFloat(FloatBuffer.wrap(kennel[c]), mKernelTex, 0, a, c, mKernelShape[0] * mKernelShape[1] + 1, 1, 1);
            }
        }
    }

    private String createShaderSource(int xSize, int ySize, int zSize) {
        String source = ShaderUtils.loadFromAssetsFile("depth_wise_conv.comp", mContext.getResources());
        return String.format(Locale.getDefault(), S_DEPTH_WISE_CONV_SHADER_HEADER, 4 * mKernelMultiplier, xSize, ySize, zSize) + source;
    }

    private void createShaderParams() {
        mParams = new int[16];
        mParams[0] = mKernelShape[0];
        mParams[1] = mKernelShape[1];
        mParams[2] = mKernelShape[2];
        mParams[3] = mKernelMultiplier;
        mParams[4] = mInShape[0];
        mParams[5] = mInShape[1];
        mParams[6] = mInShape[2];
        mParams[7] = mOutShape[0];
        mParams[8] = mOutShape[1];
        mParams[9] = mOutShape[2];
        mParams[10] = mStrides[0];
        mParams[11] = mStrides[1];
        mParams[12] = mType.index;
        mParams[13] = Utils.alignBy4(mInShape[2]);
        mParams[14] = -1 * mPadW;
        mParams[15] = -1 * mPadH;
    }

    private float[][][] createTestKernels() {
        return TestDataCreator.createDepthWiseConvKennels(mKernelShape, mKernelMultiplier);
    }

    /**
     * TODO
     * channel需要4对齐, 依次排满一个通道后再排下一个通道的值, 最后4个值为 bias, 0, 0, 0
     */
    private float[][][] loadKernels() {
        return null;
    }


    @Override
    public void initialize() {
        initConv();
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureArray(mOutTex, mAttachID, 0);
    }

    @Override
    protected void actualForwardProc(float[][] input) {
        Render.performConvolute(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mKernelTex, mNumGroupsX, mNumGroupsY, mNumGroupsZ);
    }

    @Override
    public float[] readResult() {
        float[][][] out = new float[mOutShape[2]][mOutShape[1]][mOutShape[0]];
        DataUtils.readOutput(this, mOut, mOutShape[0], mOutShape[1]);
        DataUtils.transform(out, mOut.array(), mOutShape[0], mOutShape[1], mOutShape[2], mBindLayer);
        return mOut.array();
    }
}
