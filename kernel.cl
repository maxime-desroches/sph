__kernel void density(__global const double *pos, __global double *out, __global double *out2)
{
  int i = get_global_id(0);
  int j = get_global_id(1);

  if (j <= i) {
  if (j == i) {
    out[j*1000 + i] = 0;
    out[i*1000 + j] = 0;

    out2[j*1000 + i] = 0;
    out2[i*1000 + j] = 0;
  }
  return;
  }

  double x_1 = pos[j*2];
  double y_1 = pos[j*2 + 1];

  double x_2 = pos[i*2];
  double y_2 = pos[i*2 + 1];

  double n = sqrt(pow((x_1 - x_2),2) + pow((y_1 - y_2), 2));
  double q = n/0.1;
  if(n < 0.1) {
    out[j*1000 + i] = pow((1-q), 2);
    out[i*1000 + j] = pow((1-q), 2);

    out2[j*1000 + i] = pow((1-q), 3);
    out2[i*1000 + j] = pow((1-q), 3);
  } else {
    out[j*1000 + i] = 0;
    out[i*1000 + j] = 0;

    out2[j*1000 + i] = 0;
    out2[i*1000 + j] = 0;
  }


}

__kernel void pressure(__global const double *pos, __global const double *p, __global const double *p_near, __global double *out) {
  int i = get_global_id(0)*2;
  int j = get_global_id(1)*2;

  if (j <= i) {
    if (j == i) {
      out[j*1000 + i] = 0;
      out[i*1000 + j] = 0;
      out[j*1000 + i + 1] = 0;
      out[i*1000 + j + 1] = 0;
    }
    return;
  }

  double x_1 = pos[j];
  double y_1 = pos[j + 1];

  double x_2 = pos[i];
  double y_2 = pos[i + 1];

  double n = sqrt(pow((x_1 - x_2),2) + pow((y_1 - y_2), 2));
  double q = n/0.1;

  if(n < 0.1) {
    double Q = 1-q;
    double QQ = Q*Q;
    double d = (p[i/2] + p[j/2]) * QQ + (p_near[i/2] + p_near[j/2])*QQ*Q;
    //n = clamp(n, 0.01, 0.1);
    double dx = ((x_1 - x_2)/n)*d;
    double dy = ((y_1 - y_2)/n)*d;
    out[i*1000 + j] = -dx;
    out[i*1000 + j + 1] = -dy;

    out[j*1000 + i] = dx;
    out[j*1000 + i + 1] = dy;
  } else {
    out[i*1000 + j] = 0;
    out[i*1000 + j + 1] = 0;

    out[j*1000 + i] = 0;
    out[j*1000 + i + 1] = 0;
  }


}


__kernel void viscosity(__global const double *pos, __global const double *vel, __global double *out) {
  int i = get_global_id(0)*2;
  int j = get_global_id(1)*2;

  if (j <= i) {
    if (j == i) {
      out[j*1000 + i] = 0;
      out[i*1000 + j] = 0;
      out[j*1000 + i + 1] = 0;
      out[i*1000 + j + 1] = 0;
    }
    return;
  }

  double x_1 = pos[j];
  double y_1 = pos[j + 1];

  double x_2 = pos[i];
  double y_2 = pos[i + 1];

  double n = sqrt(pow((x_1 - x_2),2) + pow((y_1 - y_2), 2));
  double q = n/0.1;

  if(n < 0.1) {
    double u = (vel[i] - vel[j])*((x_1 - x_2)/n) + (vel[i + 1] - vel[j + 1])*((y_1 - y_2)/n);

    if (u > 0.0) {
      double Ix = 0.5*(1-q)*(0.2*u)*((x_1 - x_2)/n);
      double Iy = 0.5*(1-q)*(0.2*u)*((y_1 - y_2)/n);

      out[i*1000 + j] = -Ix;
      out[i*1000 + j + 1] = -Iy;

      out[j*1000 + i] = Ix;
      out[j*1000 + i + 1] = Iy;
    } else {
      out[i*1000 + j] = 0;
      out[i*1000 + j + 1] = 0;

      out[j*1000 + i] = 0;
      out[j*1000 + i + 1] = 0;
    }

  } else {
    out[i*1000 + j] = 0;
    out[i*1000 + j + 1] = 0;

    out[j*1000 + i] = 0;
    out[j*1000 + i + 1] = 0;
  }


}
