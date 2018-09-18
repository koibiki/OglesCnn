package com.example.eglnn.layer;

import android.content.Context;
import android.text.TextUtils;

import com.example.eglnn.Render;
import com.example.eglnn.utils.Constants;
import com.example.eglnn.utils.DataUtils;
import com.example.eglnn.utils.ShaderUtils;
import com.example.eglnn.utils.Utils;

import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import static com.example.eglnn.Render.initCompPro;
import static com.example.eglnn.utils.Constants.S_CONV_GEMM_SHADER_HEADER;

/**
 *
 * */
public class ConvGEMM3 extends Layer {

    private static final String TAG = "ConvGEMM";
    private final int mKennelAmount;

    private int mNumGroupsX;

    private List<float[]> mKennels;
    private int[] mStrides;
    private int[] mKennelShape;
    private int mPad;
    private int mShaderPro;
    private int mNumGroupsZ;
    private int[] mParams;
    private int mType;
    private String mKennelFilePath;

    private int mIndexBufferId0;
    private int mIndexBufferId1;
    private int mIndexBufferId2;
    private int mIndexBufferId3;

    public ConvGEMM3(Context context, Layer preLayer, int kAmount, int k_w, int k_h, int pad, int stride_w, int stride_h, int type, String kennelFilePath) {
        super(context, preLayer);
        this.mKennelShape = new int[]{k_w, k_h, preLayer.getOutputShape()[2]};
        this.mKennelAmount = kAmount;
        this.mPad = pad;
        this.mStrides = new int[]{stride_w, stride_h};
        this.mType = type;
        this.mOutShape = calculateConvShape(kAmount);
        this.mKennelFilePath = kennelFilePath;
    }

    private int[] calculateConvShape(int kennelAmount) {
        int[] inShape = mPreLayer.getOutputShape();
        int width = calculateLength(inShape[0], mKennelShape[0], mStrides[0]);
        int height = calculateLength(inShape[1], mKennelShape[1], mStrides[1]);
        return new int[]{width, height, kennelAmount};
    }

    private int calculateLength(int length, int kennelLen, int stride) {
        return (length + 2 * mPad - kennelLen) / stride + 1;
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
        createConvInputDataIndex();

        if (TextUtils.isEmpty(mKennelFilePath)) {
            mKennels = createTestKennels();
        } else {
            mKennels = loadKennels();
        }

        createShaderParams();
    }

    private String createShaderSource(int xSize, int zSize) {
        String source = ShaderUtils.loadFromAssetsFile("conv_gemm3.comp", mContext.getResources());
        int kennelArea = mKennelShape[0] * mKennelShape[1];
        int kennelSize = kennelArea * Utils.alignBy4(mKennelShape[2]);
        return String.format(Locale.getDefault(), S_CONV_GEMM_SHADER_HEADER, kennelArea, mKennelAmount, kennelSize, xSize, 1, zSize) + source;
    }

    private void createConvInputDataIndex() {
        int outArea = mOutShape[0] * mOutShape[1];                  // int
        int kennelArea = mKennelShape[0] * mKennelShape[1];         // ivec2
        int buuferSize = Utils.alignBy4(outArea) / 4 * kennelArea * 2;
        mIndexBufferId0 = Render.initKennelBuffer(buuferSize);
        mIndexBufferId1 = Render.initKennelBuffer(buuferSize);
        mIndexBufferId2 = Render.initKennelBuffer(buuferSize);
        mIndexBufferId3 = Render.initKennelBuffer(buuferSize);

        int[][] convIndexes = new int[Utils.alignBy4(outArea)][kennelArea * 2];
        int zeroIndex = Constants.S_TEXTURE_SIZE - 1;

        for (int outX = 0; outX < mOutShape[0]; outX++) {
            for (int outY = 0; outY < mOutShape[1]; outY++) {
                for (int kX = 0; kX < mKennelShape[0]; kX++) {
                    for (int kY = 0; kY < mKennelShape[1]; kY++) {
                        int outIndex = outY * mOutShape[0] + outX;
                        int kIndex = kY * mKennelShape[0] + kX;
                        int indexXOnFeatureMap = -mPad + kX + mStrides[0] * outX;
                        int indexYOnFeatureMap = -mPad + kY + mStrides[1] * outY;
                        if (isZero(indexXOnFeatureMap, indexYOnFeatureMap)) {
                            convIndexes[outIndex][kIndex * 2] = zeroIndex;
                            convIndexes[outIndex][kIndex * 2 + 1] = zeroIndex;
                        } else {
                            convIndexes[outIndex][kIndex * 2] = indexXOnFeatureMap;
                            convIndexes[outIndex][kIndex * 2 + 1] = indexYOnFeatureMap;
                        }
                    }
                }
            }
        }
        int[] convIndexes0 = new int[Utils.alignBy4(outArea) / 4 * kennelArea * 2];
        int[] convIndexes1 = new int[Utils.alignBy4(outArea) / 4 * kennelArea * 2];
        int[] convIndexes2 = new int[Utils.alignBy4(outArea) / 4 * kennelArea * 2];
        int[] convIndexes3 = new int[Utils.alignBy4(outArea) / 4 * kennelArea * 2];

        for (int i = 0; i < Utils.alignBy4(outArea) / 4; i++) {
            System.arraycopy(convIndexes[i * 4], 0, convIndexes0, i * kennelArea * 2, kennelArea * 2);
            System.arraycopy(convIndexes[i * 4 + 1], 0, convIndexes1, i * kennelArea * 2, kennelArea * 2);
            System.arraycopy(convIndexes[i * 4 + 2], 0, convIndexes2, i * kennelArea * 2, kennelArea * 2);
            System.arraycopy(convIndexes[i * 4 + 3], 0, convIndexes3, i * kennelArea * 2, kennelArea * 2);
        }
        Render.transferToBuffer(IntBuffer.wrap(convIndexes0), mIndexBufferId0, 0);
        Render.transferToBuffer(IntBuffer.wrap(convIndexes1), mIndexBufferId1, 0);
        Render.transferToBuffer(IntBuffer.wrap(convIndexes2), mIndexBufferId2, 0);
        Render.transferToBuffer(IntBuffer.wrap(convIndexes3), mIndexBufferId3, 0);

    }

    private boolean isZero(int x, int y) {
        return (x < 0 || x >= mInShape[0]) || (y < 0 || y >= mInShape[1]);
    }

    private void createShaderParams() {
        mParams = new int[14];
        mParams[0] = mKennelShape[0];
        mParams[1] = mKennelShape[1];
        mParams[2] = mKennelShape[2];
        mParams[3] = mInShape[0];
        mParams[4] = mInShape[1];
        mParams[5] = mInShape[2];
        mParams[6] = mOutShape[0];
        mParams[7] = mOutShape[1];
        mParams[8] = mOutShape[2];
        mParams[9] = mStrides[0];
        mParams[10] = mStrides[1];
        mParams[11] = mPad;
        mParams[12] = mType;
        mParams[13] = Utils.alignBy4(mInShape[2]);
    }

    private List<float[]> createTestKennels() {
        List<float[]> kennels = new ArrayList<>();
        for (int i = 0; i < mOutShape[2]; i++) {
            kennels.add(DataUtils.createConvKennel(i + 1, mKennelShape[0], mKennelShape[1], mKennelShape[2], 1));
        }
        return kennels;
    }

    /**
     * TODO
     * channel需要4对齐, 依次排满一个通道后再排下一个通道的值, 最后4个值为 bias, 0, 0, 0
     */
    private List<float[]> loadKennels() {
        return null;
    }


    @Override
    public void initialize() {
        initConv();
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureArray(mOutTex, mAttachID);
        Render.bindBuffer(mIndexBufferId0);
        Render.bindBuffer(mIndexBufferId1);
        Render.bindBuffer(mIndexBufferId2);
        Render.bindBuffer(mIndexBufferId3);

    }

    @Override
    protected void actualForwardProc(float[][] input) {
        Render.performConvoluteGEMM3(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex,
                mIndexBufferId0, mIndexBufferId1, mIndexBufferId2, mIndexBufferId3,
                mNumGroupsX, mNumGroupsZ);
    }
}
