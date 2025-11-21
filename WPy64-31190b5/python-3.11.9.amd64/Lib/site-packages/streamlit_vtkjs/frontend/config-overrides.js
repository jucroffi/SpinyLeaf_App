module.exports = function override(config, env) {

  if (!config.resolve) {
    config.resolve = {}
  }

  if (!config.resolve.alias) {
    config.resolve.alias = {}
  }

  // when trying to use npm link to install lavender-vtkjs from the component-library
  // react throws an error that is solved by including the react alias as below
  // don't reorder this list to put react first the other aliased packages won't be resolved.
  Object.assign(config.resolve.alias, { "react/jsx-runtime": require.resolve("react/jsx-runtime.js"), "react/jsx-dev-runtime": require.resolve("react/jsx-dev-runtime.js"), "react": require.resolve("react"), })

  return config
}