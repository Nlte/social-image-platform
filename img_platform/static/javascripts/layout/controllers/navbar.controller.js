/**
* NavbarController
* @namespace platform.layout.controllers
*/
(function () {
  'use strict';

  angular
    .module('platform.layout.controllers')
    .controller('NavbarController', NavbarController);

  NavbarController.$inject = ['$scope', 'Authentication'];

  /**
  * @namespace NavbarController
  */
  function NavbarController($scope, Authentication) {
    var vm = this;

    vm.logout = logout;

    /**
    * @name logout
    * @desc Log the user out
    * @memberOf platform.layout.controllers.NavbarController
    */
    function logout() {
      Authentication.logout();
    }
  }
})();
