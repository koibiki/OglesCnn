package com.example.eglnn.layer;

import android.content.Context;

import com.example.eglnn.Render;
import com.example.eglnn.utils.MessagePackUtils;
import com.example.eglnn.utils.ShaderUtils;
import com.example.eglnn.utils.TestDataCreator;
import com.example.eglnn.utils.Utils;

import java.nio.FloatBuffer;
import java.util.Locale;

import static com.example.eglnn.Render.initCompPro;
import static com.example.eglnn.utils.Constants.S_COMMON_SHADER_HEADER;

/**
 * 使用GEMM优化的卷积
 */
public class ConvGEMM extends Layer {

    private static final String TAG = "ConvGEMM";
    private final int mKernelAmount;
    private PaddingType mPadType;

    private int mNumGroupsX;

    private float[][][] mKernels;
    private int[] mStrides;
    private int[] mKernelShape;
    private int mShaderPro;
    private int mNumGroupsZ;
    private int[] mParams;
    private ActiveType mType;
    private int mKernelTex;
    private int mPadW, mPadH;

    private ConvGEMM(Context context, String name, Layer preLayer, int kAmount, int k_w, int k_h, int stride_w, int stride_h, ActiveType type) {
        super(context, name, preLayer);
        this.mKernelShape = new int[]{k_w, k_h, preLayer.getOutputShape()[2]};
        this.mKernelAmount = kAmount;
        this.mStrides = new int[]{stride_w, stride_h};
        this.mType = type;

    }

    public ConvGEMM(Context context, String name, Layer preLayer, int kAmount, int k_w, int k_h, PaddingType padType, int stride_w, int stride_h, ActiveType type) {
        this(context, name, preLayer, kAmount, k_w, k_h, stride_w, stride_h, type);
        this.mPadType = padType;
        this.mOutShape = calculateConvShapeByType(kAmount);
    }

    public ConvGEMM(Context context, String name, Layer preLayer, int kAmount, int k_w, int k_h, int pad_w, int pad_h, int stride_w, int stride_h, ActiveType type) {
        this(context, name, preLayer, kAmount, k_w, k_h, stride_w, stride_h, type);
        this.mPadW = pad_w;
        this.mPadH = pad_h;
        this.mOutShape = calculateConvShapeByPad(kAmount);
    }

    private int[] calculateConvShapeByType(int kennelAmount) {
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
        return new int[]{width, height, kennelAmount};
    }

    private int[] calculateConvShapeByPad(int kennelAmount) {
        int width = (int) Math.ceil((mInShape[0] + 2 * mPadW - mKernelShape[0]) * 1.0f / mStrides[0]);
        int height = (int) Math.ceil((mInShape[1] + 2 * mPadH - mKernelShape[1]) * 1.0f / mStrides[1]);
        return new int[]{width, height, kennelAmount};
    }


    private int getLocalSizeX(int[] outShape) {
        int outArea = outShape[0] * outShape[1];
        if (outArea >= 1024 * 4) {
            return 1024;
        } else {
            return (int) Math.ceil(outArea * 1.0f / 4);
        }
    }

    private int getLocalSizeZ(int xSize) {
        int maxZSize = 1024 / xSize >= 64 ? 64 : 1024 / xSize;
        int zSize = Utils.alignBy4(mOutShape[2]) / 4;
        if (zSize >= maxZSize) {
            return maxZSize;
        } else {
            return zSize;
        }
    }

    private void initConv() {
        int xSize = getLocalSizeX(mOutShape);
        mNumGroupsX = (int) Math.ceil(Utils.alignBy4(mOutShape[0] * mOutShape[1]) * 1.0f / 4 / xSize);

        int zSize = getLocalSizeZ(xSize);
        mNumGroupsZ = (int) Math.ceil(Utils.alignBy4(mOutShape[2]) * 1.0f / 4 / zSize);

        String source = createShaderSource(xSize, zSize);
        mShaderPro = initCompPro(source);
        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createFloatTextureArray(mOutShape[0], mOutShape[1], Utils.alignBy4(mOutShape[2]) / 4);

//        mKernels = createTestKennels();
        mKernels = loadKennels();

        // kennel 最后一列为bias
        mKernelTex = Render.createFloatTextureArray(mKernelShape[0] * mKernelShape[1] + 1, mKernelAmount, Utils.alignBy4(mInShape[2]) / 4);

        transferToKennelTex(mKernels, mKernelTex);

        createShaderParams();
    }

    private void transferToKennelTex(float[][][] kernels, int kernelTex) {
        for (int a = 0; a < mKernelAmount; a++) {
            float[][] kennel = kernels[a];
            for (int c = 0; c < kennel.length; c++) {
                Render.transferToTextureArrayFloat(FloatBuffer.wrap(kennel[c]), kernelTex, 0, a, c, mKernelShape[0] * mKernelShape[1] + 1, 1, 1);
            }
        }
    }

    private String createShaderSource(int xSize, int zSize) {
        String shaderFile = "conv_gemm.comp";
        String source = ShaderUtils.loadFromAssetsFile(shaderFile, mContext.getResources());
        return String.format(Locale.getDefault(), S_COMMON_SHADER_HEADER , xSize, 1, zSize) + source;
    }

    private void createShaderParams() {
        mParams = new int[15];
        mParams[0] = mKernelShape[0];
        mParams[1] = mKernelShape[1];
        mParams[2] = mKernelShape[2];
        mParams[3] = mInShape[0];
        mParams[4] = mInShape[1];
        mParams[5] = mInShape[2];
        mParams[6] = mOutShape[0];
        mParams[7] = mOutShape[1];
        mParams[8] = mOutShape[2];
        mParams[9] = mStrides[0];
        mParams[10] = mStrides[1];
        mParams[11] = mType.index;
        mParams[12] = Utils.alignBy4(mInShape[2]) / 4;
        mParams[13] = -1 * mPadW;
        mParams[14] = -1 * mPadH;
    }

    private float[][][] createTestKennels() {
        return TestDataCreator.createConvKennels(mKernelShape, mKernelAmount);
    }

    private float[][][] loadKennels() {
        String kernel_file = "cifar10/" + mName;
        Object[] objects = MessagePackUtils.unpackParam(mContext, kernel_file, new Class[]{float[][][][].class, float[].class});
        if (objects != null) {
            float[][][][] kernel = (float[][][][]) objects[0];
            float[] bias = (float[]) objects[1];
            return transferKernelFormat(kernel, bias);
        } else {
            return null;
        }
    }

    /**
     * 更改kernel存放方式 amount depth (= ceil(channel / 4)) height x width x 4 + 4 (bias, 0, 0, 0) ,
     */
    private float[][][] transferKernelFormat(float[][][][] kernels, float[] bias) {
        int k_w = mKernelShape[0];
        int k_h = mKernelShape[1];
        int k_c = mKernelShape[2];
        int align_c = Utils.alignBy4(k_c);
        float[][][] new_kernenls = new float[mKernelAmount][align_c / 4][(k_w * k_h + 1) * 4];
        for (int a = 0; a < mKernelAmount; a++) {
            for (int w = 0; w < k_w; w++) {
                for (int h = 0; h < k_h; h++) {
                    for (int c = 0; c < k_c; c++) {
                        new_kernenls[a][c / 4][(w + h * k_w) * 4 + c % 4] = kernels[a][c][w][h];
                    }
                }
            }
            new_kernenls[a][0][k_w * k_h * 4] = bias[a];
        }
        return new_kernenls;
    }

    @Override
    public void initialize() {
        initConv();
    }

    @Override
    protected void bindTextureAndBuffer() {
//        Render.bindTextureArray(mOutTex, mAttachID, Utils.alignBy4(mOutShape[2]) / 4 - 1);
        Render.bindTextureArray(mOutTex, mAttachID, 0);

    }

    @Override
    protected void actualForwardProc(float[][] input) {
        Render.performConvolute(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mKernelTex, mNumGroupsX, 1, mNumGroupsZ);
    }

}
