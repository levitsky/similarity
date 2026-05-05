#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

typedef struct {
    double (*get_value)(PyArrayObject *, npy_intp);
} Accessor;

static double get_float32(PyArrayObject *array, npy_intp index) {
    return (double)*(float *)PyArray_GETPTR1(array, index);
}

static double get_float64(PyArrayObject *array, npy_intp index) {
    return *(double *)PyArray_GETPTR1(array, index);
}

static int configure_accessor(PyArrayObject **array, Accessor *accessor) {
    int typenum = PyArray_TYPE(*array);
    if (typenum == NPY_FLOAT32) {
        accessor->get_value = get_float32;
        return 0;
    }
    if (typenum == NPY_FLOAT64) {
        accessor->get_value = get_float64;
        return 0;
    }

    PyArrayObject *cast = (PyArrayObject *)PyArray_Cast(*array, NPY_FLOAT64);
    if (cast == NULL) {
        return -1;
    }
    Py_DECREF(*array);
    *array = cast;
    accessor->get_value = get_float64;
    return 0;
}

static npy_intp count_matches(
    PyArrayObject *mz1,
    PyArrayObject *mz2,
    const Accessor *accessor1,
    const Accessor *accessor2,
    double atol,
    double rtol
) {
    npy_intp n1 = PyArray_DIM(mz1, 0);
    npy_intp n2 = PyArray_DIM(mz2, 0);
    npy_intp left = 0;
    npy_intp right = 0;
    npy_intp total = 0;

    for (npy_intp i = 0; i < n1; ++i) {
        double x = accessor1->get_value(mz1, i);
        double lower = nextafter((x - atol) / (1.0 + rtol), -INFINITY);
        double upper = nextafter((x + atol) / (1.0 - rtol), INFINITY);

        while (left < n2 && accessor2->get_value(mz2, left) < lower) {
            ++left;
        }
        if (right < left) {
            right = left;
        }
        while (right < n2 && accessor2->get_value(mz2, right) <= upper) {
            ++right;
        }
        for (npy_intp j = left; j < right; ++j) {
            double y = accessor2->get_value(mz2, j);
            if (fabs(x - y) <= atol + rtol * fabs(y)) {
                ++total;
            }
        }
    }
    return total;
}

static void fill_matches(
    PyArrayObject *mz1,
    PyArrayObject *mz2,
    const Accessor *accessor1,
    const Accessor *accessor2,
    double atol,
    double rtol,
    npy_intp *idx1,
    npy_intp *idx2
) {
    npy_intp n1 = PyArray_DIM(mz1, 0);
    npy_intp n2 = PyArray_DIM(mz2, 0);
    npy_intp left = 0;
    npy_intp right = 0;
    npy_intp out = 0;

    for (npy_intp i = 0; i < n1; ++i) {
        double x = accessor1->get_value(mz1, i);
        double lower = nextafter((x - atol) / (1.0 + rtol), -INFINITY);
        double upper = nextafter((x + atol) / (1.0 - rtol), INFINITY);

        while (left < n2 && accessor2->get_value(mz2, left) < lower) {
            ++left;
        }
        if (right < left) {
            right = left;
        }
        while (right < n2 && accessor2->get_value(mz2, right) <= upper) {
            ++right;
        }
        for (npy_intp j = left; j < right; ++j) {
            double y = accessor2->get_value(mz2, j);
            if (fabs(x - y) <= atol + rtol * fabs(y)) {
                idx1[out] = i;
                idx2[out] = j;
                ++out;
            }
        }
    }
}

static PyObject *match_peaks_sorted(PyObject *self, PyObject *args) {
    PyObject *obj1;
    PyObject *obj2;
    double atol;
    double rtol;

    if (!PyArg_ParseTuple(args, "OOdd", &obj1, &obj2, &atol, &rtol)) {
        return NULL;
    }
    if (rtol >= 1.0) {
        PyErr_SetString(PyExc_ValueError, "rtol must be smaller than 1.0");
        return NULL;
    }

    PyArrayObject *mz1 = (PyArrayObject *)PyArray_FromAny(
        obj1, NULL, 1, 1, NPY_ARRAY_CARRAY_RO, NULL
    );
    PyArrayObject *mz2 = (PyArrayObject *)PyArray_FromAny(
        obj2, NULL, 1, 1, NPY_ARRAY_CARRAY_RO, NULL
    );
    if (mz1 == NULL || mz2 == NULL) {
        Py_XDECREF(mz1);
        Py_XDECREF(mz2);
        return NULL;
    }

    Accessor accessor1;
    Accessor accessor2;
    if (configure_accessor(&mz1, &accessor1) < 0 || configure_accessor(&mz2, &accessor2) < 0) {
        Py_DECREF(mz1);
        Py_DECREF(mz2);
        return NULL;
    }

    npy_intp total = count_matches(mz1, mz2, &accessor1, &accessor2, atol, rtol);
    npy_intp dims[1] = {total};
    PyArrayObject *idx1 = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INTP);
    PyArrayObject *idx2 = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INTP);
    if (idx1 == NULL || idx2 == NULL) {
        Py_DECREF(mz1);
        Py_DECREF(mz2);
        Py_XDECREF(idx1);
        Py_XDECREF(idx2);
        return NULL;
    }

    fill_matches(
        mz1,
        mz2,
        &accessor1,
        &accessor2,
        atol,
        rtol,
        (npy_intp *)PyArray_DATA(idx1),
        (npy_intp *)PyArray_DATA(idx2)
    );

    Py_DECREF(mz1);
    Py_DECREF(mz2);
    return Py_BuildValue("NN", idx1, idx2);
}

static PyObject *similarity_score(PyObject *self, PyObject *args) {
    PyObject *intensities1_obj;
    PyObject *intensities2_obj;
    PyObject *idx1_obj;
    PyObject *idx2_obj;

    if (!PyArg_ParseTuple(
            args,
            "OOOO",
            &intensities1_obj,
            &intensities2_obj,
            &idx1_obj,
            &idx2_obj)) {
        return NULL;
    }

    PyArrayObject *intensities1 = (PyArrayObject *)PyArray_FromAny(
        intensities1_obj, NULL, 1, 1, NPY_ARRAY_CARRAY_RO, NULL
    );
    PyArrayObject *intensities2 = (PyArrayObject *)PyArray_FromAny(
        intensities2_obj, NULL, 1, 1, NPY_ARRAY_CARRAY_RO, NULL
    );
    PyArrayObject *idx1 = (PyArrayObject *)PyArray_FROM_OTF(
        idx1_obj, NPY_INTP, NPY_ARRAY_CARRAY_RO
    );
    PyArrayObject *idx2 = (PyArrayObject *)PyArray_FROM_OTF(
        idx2_obj, NPY_INTP, NPY_ARRAY_CARRAY_RO
    );

    if (intensities1 == NULL || intensities2 == NULL || idx1 == NULL || idx2 == NULL) {
        Py_XDECREF(intensities1);
        Py_XDECREF(intensities2);
        Py_XDECREF(idx1);
        Py_XDECREF(idx2);
        return NULL;
    }

    if (PyArray_DIM(idx1, 0) != PyArray_DIM(idx2, 0)) {
        Py_DECREF(intensities1);
        Py_DECREF(intensities2);
        Py_DECREF(idx1);
        Py_DECREF(idx2);
        PyErr_SetString(PyExc_ValueError, "idx1 and idx2 must have the same length");
        return NULL;
    }

    Accessor intensity1_accessor;
    Accessor intensity2_accessor;
    if (configure_accessor(&intensities1, &intensity1_accessor) < 0
        || configure_accessor(&intensities2, &intensity2_accessor) < 0) {
        Py_DECREF(intensities1);
        Py_DECREF(intensities2);
        Py_DECREF(idx1);
        Py_DECREF(idx2);
        return NULL;
    }

    npy_intp n_pairs = PyArray_DIM(idx1, 0);
    npy_intp n1 = PyArray_DIM(intensities1, 0);
    npy_intp n2 = PyArray_DIM(intensities2, 0);
    npy_intp *idx1_data = (npy_intp *)PyArray_DATA(idx1);
    npy_intp *idx2_data = (npy_intp *)PyArray_DATA(idx2);

    double numerator = 0.0;
    for (npy_intp k = 0; k < n_pairs; ++k) {
        npy_intp i = idx1_data[k];
        npy_intp j = idx2_data[k];
        if (i < 0 || i >= n1 || j < 0 || j >= n2) {
            Py_DECREF(intensities1);
            Py_DECREF(intensities2);
            Py_DECREF(idx1);
            Py_DECREF(idx2);
            PyErr_SetString(PyExc_IndexError, "index out of bounds in idx1/idx2");
            return NULL;
        }
        numerator += intensity1_accessor.get_value(intensities1, i)
            * intensity2_accessor.get_value(intensities2, j);
    }

    double denom1 = 0.0;
    for (npy_intp i = 0; i < n1; ++i) {
        double x = intensity1_accessor.get_value(intensities1, i);
        denom1 += x * x;
    }

    double denom2 = 0.0;
    for (npy_intp i = 0; i < n2; ++i) {
        double x = intensity2_accessor.get_value(intensities2, i);
        denom2 += x * x;
    }

    double ndotproduct = (numerator * numerator) / denom1 / denom2;
    if (ndotproduct < -1.0) {
        ndotproduct = -1.0;
    } else if (ndotproduct > 1.0) {
        ndotproduct = 1.0;
    }
    double score = 1.0 - 2.0 * acos(ndotproduct) / M_PI;

    Py_DECREF(intensities1);
    Py_DECREF(intensities2);
    Py_DECREF(idx1);
    Py_DECREF(idx2);
    return PyFloat_FromDouble(score);
}

static PyMethodDef MatchPeaksMethods[] = {
    {
        "match_peaks_sorted",
        match_peaks_sorted,
        METH_VARARGS,
        "Return all matching peak index pairs for sorted m/z arrays."
    },
    {
        "similarity_score",
        similarity_score,
        METH_VARARGS,
        "Compute the spectrum similarity score from matched peak indices."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef matchpeaksmodule = {
    PyModuleDef_HEAD_INIT,
    "_match_peaks",
    NULL,
    -1,
    MatchPeaksMethods
};

PyMODINIT_FUNC PyInit__match_peaks(void) {
    import_array();
    return PyModule_Create(&matchpeaksmodule);
}